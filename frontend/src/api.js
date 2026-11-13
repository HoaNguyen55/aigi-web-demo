// Mọi lời gọi API nằm ở đây. Build với VITE_MOCK=1 để dùng dữ liệu mẫu (mặc định tắt).
const MOCK = import.meta.env.VITE_MOCK === '1'

export class ApiError extends Error {
  constructor(code, message, options) {
    super(message, options)
    this.code = code
  }
}

const OFFLINE = 'Không kết nối được máy chủ. Kiểm tra backend đang chạy rồi thử lại.'
const PREDICT_TIMEOUT_MS = 5 * 60 * 1000

async function request(path, init) {
  let res
  try {
    res = await fetch(path, init)
  } catch (cause) {
    if (cause.name === 'TimeoutError') {
      throw new ApiError('timeout', 'Máy chủ không trả lời trong 5 phút. Thử lại với ảnh nhỏ hơn.', { cause })
    }
    throw new ApiError('network', OFFLINE, { cause })
  }
  const text = await res.text()
  let body = null
  try {
    body = JSON.parse(text)
  } catch {
    body = null // không phải JSON (trang lỗi HTML, proxy 502...) → xử lý như phản hồi không hợp lệ bên dưới
  }
  if (!res.ok) {
    throw new ApiError(
      body?.error?.code ?? 'http_error',
      body?.error?.message ?? `Máy chủ trả về lỗi ${res.status}.`,
    )
  }
  return body
}

const isScore = (n) => Number.isFinite(n) && n >= 0 && n <= 1
const isCount = (n) => Number.isFinite(n) && n >= 0

const isModelList = (list) =>
  Array.isArray(list) &&
  list.length > 0 &&
  list.every(
    (m) =>
      typeof m?.id === 'string' &&
      typeof m.name === 'string' &&
      typeof m.description === 'string' &&
      isScore(m.threshold),
  )

const isPrediction = (r) =>
  (r?.label === 'ai' || r?.label === 'real') &&
  typeof r.model_name === 'string' &&
  [r.score_ai, r.score_real, r.threshold].every(isScore) &&
  [r.inference_ms, r.n_windows, r.image?.width, r.image?.height].every(isCount)

/** Phản hồi 200 thiếu hoặc sai trường là lỗi, không được hiển thị như kết quả hợp lệ. */
function checked(body, isValid) {
  if (!isValid(body)) {
    throw new ApiError('bad_response', 'Máy chủ trả về dữ liệu không đúng cấu trúc đã thống nhất.')
  }
  return body
}

/** GET /api/models → [{ id, name, description, threshold }] */
export async function getModels() {
  return checked(MOCK ? await mockGetModels() : await request('/api/models'), isModelList)
}

/** GET /api/health → { max_side }: cạnh dài tối đa (px) mà máy chủ chấp nhận */
export async function getLimits() {
  if (MOCK) return { max_side: 2048 }
  return checked(await request('/api/health'), (b) => Number.isFinite(b?.max_side) && b.max_side > 0)
}

/** POST /api/predict (multipart: file, model_id) */
export async function predict(file, modelId) {
  if (MOCK) return checked(await mockPredict(file, modelId), isPrediction)
  const form = new FormData()
  form.append('file', file)
  form.append('model_id', modelId)
  const init = { method: 'POST', body: form, signal: AbortSignal.timeout(PREDICT_TIMEOUT_MS) }
  return checked(await request('/api/predict', init), isPrediction)
}

// Dữ liệu mẫu, chỉ dùng khi VITE_MOCK=1. Tên file quyết định kết quả: chứa "real" → Real Image,
// "fail" → inference_failed, "offline" → mất kết nối, "slow" → chờ 8 giây, còn lại → AI-Generated.

const MOCK_MODELS = [
  { id: 'rgb', name: 'ViT + RGB', description: 'Giá trị màu gốc (đường cơ sở)', threshold: 0.887 },
  { id: 'npr', name: 'ViT + NPR', description: 'Chênh lệch giữa các điểm ảnh lân cận', threshold: 0.7236 },
  { id: 'hh', name: 'ViT + HH', description: 'Băng chéo wavelet Haar (nhiễu tần số cao)', threshold: 0.6874 },
  { id: 'wpt', name: 'ViT + WPT', description: '16 dải tần wavelet packet', threshold: 0.5413 },
]

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function mockGetModels() {
  await delay(200)
  return MOCK_MODELS
}

async function mockPredict(file, modelId) {
  const name = file.name.toLowerCase()
  await delay(name.includes('slow') ? 8000 : 1200)
  if (name.includes('offline')) throw new ApiError('network', OFFLINE)
  if (name.includes('fail')) throw new ApiError('inference_failed', 'Mô hình gặp lỗi khi suy luận (dữ liệu mẫu).')
  const model = MOCK_MODELS.find((m) => m.id === modelId)
  const scoreAi = name.includes('real') ? 0.07 : 0.93
  return {
    model_id: model.id,
    model_name: model.name,
    label: scoreAi >= model.threshold ? 'ai' : 'real',
    score_ai: scoreAi,
    score_real: 1 - scoreAi,
    threshold: model.threshold,
    inference_ms: 1840,
    image: { width: 512, height: 512 },
    n_windows: 9,
  }
}
