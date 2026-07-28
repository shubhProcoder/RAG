/**
 * Upload component — drag-and-drop + click-to-browse with
 * progress indication and validation.
 */

const Upload = (() => {
    const MAX_SIZE_MB = 50;
    const ALLOWED_TYPE = 'application/pdf';

    let _zone, _input, _content, _progress, _progressFill, _progressText;
    let _onComplete = null;

    // ── Init ────────────────────────────────────────────────────────

    function init(onComplete) {
        _onComplete = onComplete;

        _zone         = document.getElementById('upload-zone');
        _input        = document.getElementById('file-input');
        _content      = document.getElementById('upload-content');
        _progress     = document.getElementById('upload-progress');
        _progressFill = document.getElementById('progress-fill');
        _progressText = document.getElementById('progress-text');

        // Click to browse
        _zone.addEventListener('click', () => _input.click());
        _zone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); _input.click(); }
        });
        _input.addEventListener('change', () => {
            if (_input.files.length) _handleFiles(_input.files);
        });

        // Drag & drop
        _zone.addEventListener('dragover',  _onDragOver);
        _zone.addEventListener('dragleave', _onDragLeave);
        _zone.addEventListener('drop',      _onDrop);
    }

    // ── Drag events ─────────────────────────────────────────────────

    function _onDragOver(e) {
        e.preventDefault();
        _zone.classList.add('upload-zone--active');
    }

    function _onDragLeave(e) {
        e.preventDefault();
        _zone.classList.remove('upload-zone--active');
    }

    function _onDrop(e) {
        e.preventDefault();
        _zone.classList.remove('upload-zone--active');
        const files = e.dataTransfer.files;
        if (files.length) _handleFiles(files);
    }

    // ── File handling ───────────────────────────────────────────────

    async function _handleFiles(fileList) {
        const files = Array.from(fileList);

        // Validate
        const invalid = [];
        const valid = [];

        for (const file of files) {
            if (file.type !== ALLOWED_TYPE && !file.name.toLowerCase().endsWith('.pdf')) {
                invalid.push(`"${file.name}" is not a PDF`);
            } else if (file.size > MAX_SIZE_MB * 1024 * 1024) {
                invalid.push(`"${file.name}" exceeds ${MAX_SIZE_MB}MB limit`);
            } else {
                valid.push(file);
            }
        }

        if (invalid.length) {
            App.toast(invalid.join('. '), 'error');
        }

        if (!valid.length) return;

        // Show progress
        _showProgress();

        try {
            const result = await API.uploadFiles(valid, (pct) => {
                _updateProgress(pct, `Uploading… ${pct}%`);
            });

            _updateProgress(100, 'Processing…');
            App.toast(result.message || `Uploaded ${valid.length} file(s)`, 'success');

            if (_onComplete) _onComplete(result);
        } catch (err) {
            App.toast(err.message || 'Upload failed', 'error');
        } finally {
            // Reset after a short delay
            setTimeout(_hideProgress, 1200);
            _input.value = '';
        }
    }

    // ── Progress UI ─────────────────────────────────────────────────

    function _showProgress() {
        _content.style.display = 'none';
        _progress.hidden = false;
        _progressFill.style.width = '0%';
    }

    function _updateProgress(pct, text) {
        _progressFill.style.width = `${pct}%`;
        _progressText.textContent = text;
    }

    function _hideProgress() {
        _progress.hidden = true;
        _content.style.display = '';
        _progressFill.style.width = '0%';
    }

    return { init };
})();
