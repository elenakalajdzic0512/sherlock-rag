# Phase 7a: frozen question generation protocol

Recorded before the first real question-generation call. This is a production-content batch from the verified book, not an independent evaluation benchmark.

## Fixed inputs and selection

- Use the existing verified `w250_o50` (A) chunks, without retrieval or the embedding model. Group chunks by the twelve source stories in their existing order. For each story with `n` chunks, choose zero-based positions `floor((n-1)*f)` for `f = 0.25, 0.50, 0.75`, in that order. Do not select by prior evaluation questions or output quality.
- The exact 36 selected IDs, positions and story titles were frozen in `questions/generated_v1/selection.json` before model loading; SHA-256 `6d10a4bf2d0ce1ab9047f6982f1b6efd31be165ca1e103f2c161d95213d383c0`.
- Prompt `prompts/question_generation_v1.txt`, SHA-256 `27d70824e32b508226d61480d5095a86416826b685f2109e29cc07fbb49db23f`. Config `config/question_generation_v1.json`, SHA-256 `fe346dbf189c3a09aa9b16e795be3f418824a0ead5ba4362c41897aed2b45e6a`.
- The full runtime fingerprint before model loading is `27d12172fa3605184cf8f3381493986737006e7534b1898ba91407773f4d284c`. It includes the selected IDs, source/chunk hashes, prompt/config hashes, pinned model revision, package versions and deterministic hashes of all package Python source.

## Generation and validation

Use the existing pinned `Qwen/Qwen3-4B-Instruct-2507` revision `cdbee75f17c01a7cc42f958dc650907174af0554`, bfloat16 MPS, greedy decoding, complete-chat input limit 6144 and at most 512 output tokens per attempt. Load the model once for the batch. Send only one normalized, verified C1 passage per call, with its story title and chunk ID. Do not send previous evaluation questions, expected answers or gold intervals.

For each passage, accept at most one candidate or a distinct `insufficient_evidence` outcome. The six-field JSON contract is exact; a candidate requires a nonempty standalone question, short expected answer, source ID C1, exact contiguous displayed quote, and empty reason. `insufficient_evidence` requires empty candidate fields and a nonempty reason. Check full chat token length without truncation. Permit one mechanical repair attempt using the same passage; save every raw attempt and validation error. Resolve the quote to the book with existing citation logic; the model never supplies offsets. Mark duplicate normalized question texts; do not discard their raw record.

Checkpoint after each of the 36 selections. Resume only when run ID, selected prefix and full fingerprint match. Lock the complete raw `results.json` with `results.sha256` before content review. A completed run is read without model loading or regeneration. Separately review every mechanically valid candidate against its sent passage, with explicit rationale for premise, answer, support, quote sufficiency and quote precision. The reviewed bank includes accepted, nonduplicate questions only. Do not invent replacements for unproductive passages.
