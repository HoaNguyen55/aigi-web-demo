import { useEffect, useRef, useState } from 'react'
import { getLimits, getModels, predict } from './api.js'
import { validateFile } from './validate.js'

const ACCEPT = '.jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp'

const ERROR_TITLES = {
  unsupported_type: 'Định dạng không được hỗ trợ',
  file_too_large: 'Ảnh quá lớn',
  invalid_image: 'Ảnh không hợp lệ',
  inference_failed: 'Lỗi suy luận',
  unknown_model: 'Mô hình không còn hợp lệ',
  busy: 'Máy chủ đang bận',
  http_error: 'Lỗi máy chủ',
  bad_response: 'Phản hồi không hợp lệ',
  timeout: 'Quá thời gian chờ',
  network: 'Mất kết nối',
}
const FILE_ERRORS = ['unsupported_type', 'file_too_large', 'invalid_image']
const SIDE_CHOICES = [512, 1024, 2048, 4096]

const num = (n, digits) =>
  n.toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
const fileSize = (bytes) => (bytes < 1024 * 1024 ? `${num(bytes / 1024, 1)} KB` : `${num(bytes / 1024 / 1024, 2)} MB`)
const pct = (n) => `${num(n * 100, 1)}%`

const card = 'rounded-lg border border-border bg-bg'
const focus = 'focus-visible:outline-none focus-visible:shadow-focus'
const btnSecondary = `rounded-md border border-border bg-bg px-3 py-1.5 text-sm font-medium text-fg transition-colors duration-150 hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50 ${focus}`

function Dropzone({ file, disabled, inputRef, onPick }) {
  const [over, setOver] = useState(false)
  const drop = (e) => {
    e.preventDefault()
    setOver(false)
    if (!disabled) onPick(e.dataTransfer.files[0])
  }
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      className={`relative flex min-h-36 min-w-0 flex-col items-center justify-center rounded-lg border border-dashed p-4 text-center transition-colors duration-150 focus-within:shadow-focus xl:min-h-0 xl:flex-1 ${
        over ? 'border-accent bg-accent-soft' : 'border-border bg-surface'
      }`}
    >
      <input
        ref={inputRef}
        id="file-input"
        type="file"
        accept={ACCEPT}
        disabled={disabled}
        className="sr-only"
        tabIndex={file ? -1 : undefined}
        aria-hidden={file ? true : undefined}
        onChange={(e) => {
          onPick(e.target.files[0])
          e.target.value = ''
        }}
      />
      <label
        htmlFor="file-input"
        className={`absolute inset-0 rounded-lg ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <span className="sr-only">Chọn ảnh để phân tích</span>
      </label>
      <div className="pointer-events-none relative flex w-full min-w-0 flex-col items-center gap-2">
        {file ? (
          <>
            <p className="max-w-full truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted">{fileSize(file.size)}</p>
            <button
              type="button"
              className={`pointer-events-auto ${btnSecondary}`}
              disabled={disabled}
              onClick={() => inputRef.current.click()}
            >
              Đổi ảnh
            </button>
          </>
        ) : (
          <>
            <p className="text-sm font-medium text-accent">Kéo thả ảnh vào đây hoặc bấm để chọn</p>
            <p className="text-xs text-muted">JPG, JPEG, PNG, WEBP · tối đa 10 MB</p>
          </>
        )}
      </div>
    </div>
  )
}

function ModelPicker({ models, error, value, disabled, onChange, onRetry }) {
  return (
    <fieldset disabled={disabled} className="min-w-0">
      <legend className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">Mô hình</legend>
      {error && (
        <div role="alert" className="rounded-lg border border-danger bg-danger-soft p-3 text-sm">
          <p className="font-medium text-danger">Không tải được danh sách mô hình</p>
          <p className="mt-1 max-h-16 overflow-y-auto wrap-anywhere text-fg-2">{error.message}</p>
          <button type="button" className={`mt-2 ${btnSecondary}`} onClick={onRetry}>
            Tải lại
          </button>
        </div>
      )}
      {!error && !models && (
        <output className="block text-sm text-muted">Đang tải danh sách mô hình…</output>
      )}
      <div className="flex flex-col gap-2">
        {models?.map((m) => (
          <label
            key={m.id}
            className={`flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2 transition-colors duration-150 has-focus-visible:shadow-focus ${
              value === m.id ? 'border-accent bg-accent-soft' : 'border-border hover:border-accent'
            }`}
          >
            <input
              type="radio"
              name="model"
              value={m.id}
              checked={value === m.id}
              onChange={() => onChange(m.id)}
              className="mt-1 accent-accent focus:outline-none"
            />
            <span className="min-w-0 wrap-anywhere">
              <span className="block text-sm font-medium">{m.name}</span>
              <span className="block text-xs text-muted">{m.description}</span>
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

function Options({ sides, side, onSide, thrText, thrInvalid, thrHint, onThr, runAll, onRunAll, disabled }) {
  const field = `mt-1 block h-9 w-full rounded-md border bg-bg px-2 text-sm text-fg ${focus}`
  return (
    <fieldset disabled={disabled} className="grid min-w-0 shrink-0 grid-cols-2 gap-x-3 gap-y-2">
      <legend className="sr-only">Tuỳ chọn phân tích</legend>
      <label className="min-w-0 text-xs text-muted">
        Giới hạn cạnh ảnh
        <select className={`${field} border-border`} value={side ?? ''} onChange={(e) => onSide(Number(e.target.value))}>
          {sides.map((s) => (
            <option key={s} value={s}>
              {s} px
            </option>
          ))}
        </select>
      </label>
      <label className="min-w-0 text-xs text-muted">
        Ngưỡng quyết định
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          inputMode="decimal"
          value={thrText}
          placeholder={thrHint}
          aria-invalid={thrInvalid}
          onChange={(e) => onThr(e.target.value)}
          className={`${field} ${thrInvalid ? 'border-danger' : 'border-border'}`}
        />
      </label>
      {thrInvalid && (
        <p role="alert" className="col-span-2 text-xs text-danger">
          Ngưỡng phải nằm trong khoảng 0 đến 1; đang dùng ngưỡng mặc định của mô hình.
        </p>
      )}
      <label className="col-span-2 flex cursor-pointer items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={runAll}
          onChange={(e) => onRunAll(e.target.checked)}
          className={`size-4 accent-accent ${focus}`}
        />
        Chạy cả 4 mô hình và so sánh
      </label>
    </fieldset>
  )
}

function Spinner() {
  return (
    <span
      aria-hidden="true"
      className="inline-block size-4 shrink-0 animate-spin rounded-full border-2 border-current border-t-transparent"
    />
  )
}

function ScoreBar({ score, threshold }) {
  let labelSide = 'left-1/2 -translate-x-1/2'
  if (threshold > 0.8) labelSide = 'right-0'
  if (threshold < 0.2) labelSide = 'left-0'
  return (
    <div className="pt-6">
      <meter className="sr-only" aria-label="Điểm AI" min={0} max={1} value={score} />
      <div aria-hidden="true" className="relative h-2 rounded-full bg-border-soft">
        <div className="h-full rounded-full bg-accent" style={{ width: `${score * 100}%` }} />
        <div className="absolute -top-1.5 h-5 w-0.5 bg-fg" style={{ left: `${threshold * 100}%` }}>
          <span className={`absolute bottom-full text-xs whitespace-nowrap text-fg-2 ${labelSide}`}>
            ngưỡng {num(threshold, 4)}
          </span>
        </div>
      </div>
      <div className="mt-1 flex justify-between text-xs text-muted">
        <span>0</span>
        <span>1</span>
      </div>
    </div>
  )
}

function ResultCard({ result, customThr }) {
  const threshold = customThr ?? result.threshold
  const isAi = result.score_ai >= threshold
  const meta = [
    ['Mô hình', result.model_name],
    ['Ngưỡng quyết định', `${num(threshold, 4)}${customThr === null ? '' : ' (tuỳ chỉnh)'}`],
    ['Thời gian suy luận', `${Math.round(result.inference_ms)} ms`],
    ['Kích thước ảnh', `${result.image.width} × ${result.image.height} px`],
    ['Số cửa sổ đã chấm', result.n_windows],
  ]
  return (
    <output className="grid min-w-0 gap-x-8 gap-y-4 md:grid-cols-2">
      <div className="min-w-0">
        <p className="text-xs font-semibold tracking-wide text-muted uppercase">Kết quả</p>
        <p className={`text-2xl leading-tight font-semibold tracking-tight ${isAi ? 'text-accent' : 'text-success'}`}>
          {isAi ? 'AI-Generated' : 'Real Image'}
        </p>
        <p className="mt-2 flex items-baseline justify-between text-sm">
          <span className="text-fg-2">Điểm AI</span>
          <span className="text-lg font-medium tabular-nums">{num(result.score_ai, 4)}</span>
        </p>
        <p className="text-xs text-muted">Điểm AI là đầu ra của mô hình, không phải độ chính xác.</p>
        <ScoreBar score={result.score_ai} threshold={threshold} />
        <p className="mt-1 flex justify-between text-sm text-fg-2">
          <span>
            P(AI) <span className="font-medium text-fg tabular-nums">{pct(result.score_ai)}</span>
          </span>
          <span>
            P(Real) <span className="font-medium text-fg tabular-nums">{pct(result.score_real)}</span>
          </span>
        </p>
      </div>
      <dl className="min-w-0 self-center text-sm">
        {meta.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 border-b border-border-soft py-1.5">
            <dt className="text-muted">{k}</dt>
            <dd className="min-w-0 text-right font-medium wrap-anywhere">{v}</dd>
          </div>
        ))}
      </dl>
    </output>
  )
}

function CompareTable({ results, customThr }) {
  return (
    <div role="status" className="min-w-0">
      <p className="text-xs font-semibold tracking-wide text-muted uppercase">
        Kết quả · {results.length} mô hình{customThr === null ? '' : ` · ngưỡng tuỳ chỉnh ${num(customThr, 4)}`}
      </p>
      <table className="mt-2 w-full text-sm">
        <thead>
          <tr className="text-xs text-muted">
            <th scope="col" className="py-1 text-left font-medium">Mô hình</th>
            <th scope="col" className="text-left font-medium">Kết luận</th>
            <th scope="col" className="text-right font-medium">Điểm AI</th>
            <th scope="col" className="text-right font-medium">Ngưỡng</th>
            <th scope="col" className="text-right font-medium">Thời gian</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => {
            const threshold = customThr ?? r.threshold
            const isAi = r.score_ai >= threshold
            return (
              <tr key={r.model_name} className="border-t border-border-soft">
                <th scope="row" className="py-1.5 text-left font-medium">{r.model_name}</th>
                <td className={`font-semibold ${isAi ? 'text-accent' : 'text-success'}`}>
                  {isAi ? 'AI-Generated' : 'Real Image'}
                </td>
                <td className="text-right tabular-nums">{num(r.score_ai, 4)}</td>
                <td className="text-right tabular-nums">{num(threshold, 4)}</td>
                <td className="text-right tabular-nums">{Math.round(r.inference_ms)} ms</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted">
        Điểm AI là đầu ra của mô hình, không phải độ chính xác. Ảnh {results[0].image.width} × {results[0].image.height} px,{' '}
        {results[0].n_windows} cửa sổ.
      </p>
    </div>
  )
}

function ErrorBox({ error, actionLabel, onAction }) {
  return (
    <div role="alert" className="flex w-full min-w-0 flex-col items-start gap-2">
      <p className="text-lg font-semibold text-danger">{ERROR_TITLES[error.code] ?? 'Đã xảy ra lỗi'}</p>
      <p className="max-h-20 w-full overflow-y-auto text-sm wrap-anywhere text-fg-2">{error.message}</p>
      <p className="max-w-full truncate text-xs text-muted">Mã lỗi: {error.code}</p>
      <button type="button" className={btnSecondary} onClick={onAction}>
        {actionLabel}
      </button>
    </div>
  )
}

function ResultPanel({ ref, loading, progress, error, results, customThr, hasFile, actionLabel, onAction }) {
  let body = (
    <p className="text-sm text-muted">
      {hasFile
        ? 'Ảnh đã sẵn sàng. Chọn mô hình rồi bấm “Analyze Image”.'
        : 'Chưa có kết quả. Tải một ảnh lên để bắt đầu.'}
    </p>
  )
  if (results?.length === 1) body = <ResultCard result={results[0]} customThr={customThr} />
  if (results?.length > 1) body = <CompareTable results={results} customThr={customThr} />
  if (error) body = <ErrorBox error={error} actionLabel={actionLabel} onAction={onAction} />
  if (loading) {
    body = (
      <output className="block text-sm text-fg-2">
        <span className="flex items-center gap-2">
          <span className="text-accent">
            <Spinner />
          </span>
          Đang phân tích ảnh…{progress && ` (mô hình ${progress})`}
        </span>
        <span className="mt-1 block text-xs text-muted">
          Mô hình chạy trên CPU: ảnh nhỏ mất vài giây, ảnh lớn có thể mất vài chục giây đến vài phút.
        </span>
      </output>
    )
  }
  return (
    <section
      ref={ref}
      tabIndex={-1}
      aria-label="Kết quả phân tích"
      className={`${card} flex min-h-40 min-w-0 flex-col justify-center overflow-hidden p-5 focus:outline-none focus-visible:shadow-focus xl:h-64 xl:shrink-0`}
    >
      {body}
    </section>
  )
}

function Preview({ file, url, broken, onBroken, onSize }) {
  let body = <p className="text-sm text-muted">Ảnh xem trước sẽ hiện ở đây</p>
  if (file && broken) {
    body = (
      <p role="alert" className="px-4 text-center text-sm text-danger">
        Trình duyệt không hiển thị được ảnh này. File có thể hỏng hoặc không phải ảnh.
      </p>
    )
  } else if (file) {
    body = (
      <img
        src={url}
        alt={`Ảnh xem trước: ${file.name}`}
        onError={onBroken}
        onLoad={(e) => onSize({ width: e.target.naturalWidth, height: e.target.naturalHeight })}
        className="absolute inset-0 size-full object-contain p-3"
      />
    )
  }
  return (
    <section
      aria-label="Ảnh xem trước"
      className={`${card} relative flex h-72 min-w-0 items-center justify-center overflow-hidden bg-surface xl:h-auto xl:min-h-0 xl:flex-1`}
    >
      {body}
    </section>
  )
}

export default function App() {
  const [models, setModels] = useState(null)
  const [modelsError, setModelsError] = useState(null)
  const [modelId, setModelId] = useState('')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [previewBroken, setPreviewBroken] = useState(false)
  const [rejected, setRejected] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [maxSide, setMaxSide] = useState(null)
  const [sideLimit, setSideLimit] = useState(null)
  const [dims, setDims] = useState(null)
  const [thrText, setThrText] = useState('')
  const [runAll, setRunAll] = useState(false)
  const [progress, setProgress] = useState(null)
  const inputRef = useRef(null)
  const resultRef = useRef(null)

  function loadModels() {
    Promise.all([getModels(), getLimits()])
      .then(([list, limits]) => {
        setModels(list)
        setModelId((id) => (list.some((m) => m.id === id) ? id : list[0].id))
        setMaxSide(limits.max_side)
        setSideLimit((side) => (side && side <= limits.max_side ? side : limits.max_side))
      })
      .catch(setModelsError)
  }
  useEffect(loadModels, [])

  useEffect(() => {
    const block = (e) => e.preventDefault()
    window.addEventListener('dragover', block)
    window.addEventListener('drop', block)
    return () => {
      window.removeEventListener('dragover', block)
      window.removeEventListener('drop', block)
    }
  }, [])

  // Trên tablet/mobile khu kết quả nằm dưới màn hình: cuộn tới khi bắt đầu và khi có kết quả/lỗi.
  // Nút "Thử lại" biến mất khi bắt đầu phân tích; đừng để focus bàn phím rơi về <body>.
  useEffect(() => {
    if (!loading && !results && !error) return
    const panel = resultRef.current
    if (document.activeElement === document.body) panel.focus({ preventScroll: true })
    panel.scrollIntoView({ block: 'nearest' })
  }, [loading, results, error])

  function reloadModels() {
    setModelsError(null)
    setError(null)
    loadModels()
  }

  function pickFile(next) {
    if (!next) return
    const invalid = validateFile(next)
    setRejected(invalid)
    if (invalid) return
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(next))
    setPreviewBroken(false)
    setDims(null)
    setResults(null)
    setError(null)
    setFile(next)
  }

  function pickModel(id) {
    setModelId(id)
    setResults(null)
    setError(null)
  }

  function toggleRunAll(on) {
    setRunAll(on)
    setResults(null)
    setError(null)
  }

  const thrValue = thrText.trim() === '' ? null : Number(thrText)
  const thrInvalid = thrValue !== null && !(thrValue >= 0 && thrValue <= 1)
  const customThr = thrInvalid ? null : thrValue
  const sides = maxSide ? [...SIDE_CHOICES.filter((s) => s < maxSide), maxSide] : []
  const current = models?.find((m) => m.id === modelId)
  let thrHint = current ? `mặc định ${num(current.threshold, 4)}` : ''
  if (runAll) thrHint = 'theo từng mô hình'

  async function analyze() {
    if (loading) return
    setError(null)
    setResults(null)
    if (dims && sideLimit && Math.max(dims.width, dims.height) > sideLimit) {
      setError({
        code: 'file_too_large',
        message: `Ảnh ${dims.width} × ${dims.height} px vượt giới hạn cạnh dài ${sideLimit} px đang chọn. Tăng giới hạn hoặc chọn ảnh nhỏ hơn.`,
      })
      return
    }
    setLoading(true)
    const ids = runAll ? models.map((m) => m.id) : [modelId]
    const done = []
    try {
      for (const id of ids) {
        if (ids.length > 1) setProgress(`${done.length + 1}/${ids.length}`)
        done.push(await predict(file, id))
      }
      setResults(done)
    } catch (e) {
      setError({ code: e.code ?? 'unknown', message: e.message })
    } finally {
      setLoading(false)
      setProgress(null)
    }
  }

  let action = ['Thử lại', analyze]
  if (error?.code === 'unknown_model') action = ['Tải lại danh sách mô hình', reloadModels]
  if (FILE_ERRORS.includes(error?.code)) action = ['Chọn ảnh khác', () => inputRef.current.click()]

  return (
    <div className="flex min-h-screen flex-col xl:h-screen">
      <header className="flex items-baseline gap-3 border-b border-border px-4 py-3 md:px-6">
        <h1 className="text-lg font-semibold tracking-tight">Phát hiện ảnh do AI tạo</h1>
        <p className="text-xs text-muted">Demo học thuật</p>
      </header>

      <main className="grid flex-1 grid-cols-[minmax(0,1fr)] gap-6 p-4 md:p-6 xl:min-h-0 xl:grid-cols-[400px_minmax(0,1fr)]">
        <section aria-label="Đầu vào" className="flex min-h-0 min-w-0 flex-col gap-4">
          <Dropzone file={file} disabled={loading} inputRef={inputRef} onPick={pickFile} />
          {rejected && (
            <p
              role="alert"
              className="max-h-16 shrink-0 overflow-y-auto rounded-md border border-danger bg-danger-soft px-3 py-2 text-xs wrap-anywhere text-danger"
            >
              {rejected.message}
              {file && ' Ảnh đang chọn được giữ nguyên.'}
            </p>
          )}
          <ModelPicker
            models={models}
            error={modelsError}
            value={modelId}
            disabled={loading || runAll}
            onChange={pickModel}
            onRetry={reloadModels}
          />
          {models && (
            <Options
              sides={sides}
              side={sideLimit}
              onSide={setSideLimit}
              thrText={thrText}
              thrInvalid={thrInvalid}
              thrHint={thrHint}
              onThr={setThrText}
              runAll={runAll}
              onRunAll={toggleRunAll}
              disabled={loading}
            />
          )}
          <button
            type="button"
            disabled={!file || !modelId}
            aria-disabled={loading}
            aria-busy={loading}
            onClick={analyze}
            className={`flex h-11 shrink-0 items-center justify-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-accent-on transition-colors duration-150 hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-border disabled:text-muted aria-disabled:cursor-not-allowed aria-disabled:bg-border aria-disabled:text-muted ${focus}`}
          >
            {loading && <Spinner />}
            {loading ? 'Đang phân tích…' : 'Analyze Image'}
          </button>
        </section>

        <div className="flex min-h-0 min-w-0 flex-col gap-4">
          <Preview
            file={file}
            url={previewUrl}
            broken={previewBroken}
            onBroken={() => setPreviewBroken(true)}
            onSize={setDims}
          />
          <ResultPanel
            ref={resultRef}
            loading={loading}
            progress={progress}
            error={error}
            results={results}
            customThr={customThr}
            hasFile={Boolean(file)}
            actionLabel={action[0]}
            onAction={action[1]}
          />
        </div>
      </main>
    </div>
  )
}
