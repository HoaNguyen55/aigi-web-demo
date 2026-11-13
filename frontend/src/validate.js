export const MAX_BYTES = 10 * 1024 * 1024
const TYPES = ['image/jpeg', 'image/png', 'image/webp']
const EXT = /\.(jpe?g|png|webp)$/i

const megabytes = (bytes) =>
  (bytes / 1024 / 1024).toLocaleString('vi-VN', { minimumFractionDigits: 1, maximumFractionDigits: 1 })

/**
 * Trả về { code, message } nếu file không hợp lệ, ngược lại null.
 * Chấp nhận khi MIME hoặc đuôi file hợp lệ (trình duyệt có thể báo MIME lạ); server kiểm tra lại bằng nội dung.
 */
export function validateFile(file) {
  const reject = (code, why) => ({ code, message: `Không nhận “${file.name}”: ${why}` })
  if (!TYPES.includes(file.type) && !EXT.test(file.name)) {
    return reject('unsupported_type', 'chỉ nhận ảnh JPG, JPEG, PNG hoặc WEBP.')
  }
  if (file.size === 0) return reject('empty_file', 'file rỗng (0 byte).')
  if (file.size > MAX_BYTES) {
    return reject('file_too_large', `file nặng ${megabytes(file.size)} MB, vượt giới hạn 10 MB.`)
  }
  return null
}
