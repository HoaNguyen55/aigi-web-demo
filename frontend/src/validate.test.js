import test from 'node:test'
import assert from 'node:assert/strict'
import { validateFile, MAX_BYTES } from './validate.js'

test('validateFile', () => {
  assert.equal(validateFile({ name: 'a.png', type: 'image/png', size: 1 }), null)
  assert.equal(validateFile({ name: 'a.JPG', type: '', size: MAX_BYTES }), null)
  assert.equal(validateFile({ name: 'a.png', type: 'application/octet-stream', size: 1 }), null)
  assert.equal(validateFile({ name: 'a.jpg', type: 'image/pjpeg', size: 1 }), null)
  assert.equal(validateFile({ name: 'blob', type: 'image/webp', size: 1 }), null)
  assert.equal(validateFile({ name: 'a.gif', type: 'image/gif', size: 1 }).code, 'unsupported_type')
  assert.equal(validateFile({ name: 'a.txt', type: '', size: 1 }).code, 'unsupported_type')
  assert.equal(validateFile({ name: 'a.png', type: 'image/png', size: 0 }).code, 'empty_file')

  const big = validateFile({ name: 'to.webp', type: 'image/webp', size: 11.5 * 1024 * 1024 })
  assert.equal(big.code, 'file_too_large')
  assert.match(big.message, /“to\.webp”/)
  assert.match(big.message, /11,5 MB/)
})
