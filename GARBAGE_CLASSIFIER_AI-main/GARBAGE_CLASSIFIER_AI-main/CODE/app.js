document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const classifyBtn = document.getElementById('classify-btn');
    const btnText = classifyBtn.querySelector('.btn-text');
    const btnIcon = classifyBtn.querySelector('i');
    const spinner = classifyBtn.querySelector('.spinner');
    const resultSection = document.getElementById('result-section');
    
    // Result elements
    const confidenceBadge = document.getElementById('confidence-badge');
    const resultIconWrapper = document.getElementById('result-icon-wrapper');
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultDesc = document.getElementById('result-desc');
    const recommendationText = document.getElementById('recommendation-text');
    const actionRecommendation = document.querySelector('.action-recommendation');

    let currentFile = null;

    // Click to browse
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        if (!currentFile) fileInput.click();
    });

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn một tệp hình ảnh (JPG, PNG, WEBP).');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('Kích thước tệp quá lớn. Vui lòng chọn ảnh dưới 5MB.');
            return;
        }

        currentFile = file;
        
        // Hide results if showing
        resultSection.classList.add('hidden');
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        uploadContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        classifyBtn.disabled = true;
        resultSection.classList.add('hidden');
    });

    // Classification Action
    classifyBtn.addEventListener('click', () => {
        if (!currentFile) return;

        // Bật trạng thái Loading
        classifyBtn.disabled = true;
        btnText.textContent = 'Đang Phân Tích...';
        btnIcon.classList.add('hidden');
        spinner.classList.remove('hidden');
        resultSection.classList.add('hidden');

        // CHÚ Ý CHO DEVELOPER:
        // Đoạn code này chỉ là Front-end thuần túy (không có tác vụ Backend).
        // Bạn có thể viết code fetch API gọi lên server Python của bạn tại đây
        
        setTimeout(() => {
            // Giả lập giao diện kết quả sau 1 giây
            confidenceBadge.textContent = `98.5%`;
            confidenceBadge.style.color = 'var(--color-recycle)';
            confidenceBadge.style.borderColor = 'var(--color-recycle)';
            confidenceBadge.style.background = `rgba(59, 130, 246, 0.2)`;
            
            resultIconWrapper.style.background = `rgba(59, 130, 246, 0.2)`;
            resultIconWrapper.style.color = 'var(--color-recycle)';
            resultIcon.className = `fa-solid fa-recycle`;
            
            resultTitle.textContent = 'Kết quả mẫu';
            resultTitle.style.color = 'var(--color-recycle)';
            
            resultDesc.textContent = 'Giao diện Web đã sẵn sàng. Hãy kết nối nút này với API Python của bạn.';
            recommendationText.textContent = '...';
            actionRecommendation.style.borderLeftColor = 'var(--color-recycle)';

            resultSection.classList.remove('hidden');
            
            // Tắt trạng thái Loading
            classifyBtn.disabled = false;
            btnText.textContent = 'Phân Loại Lại';
            btnIcon.classList.remove('hidden');
            spinner.classList.add('hidden');
            
            if (window.innerWidth < 600) {
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }, 1000);
    });
});
