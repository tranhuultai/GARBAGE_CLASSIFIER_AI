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

    // Định nghĩa thông tin chi tiết cho các danh mục rác thải
    const categoryInfo = {
        'recycle': {
            name: 'Rác Tái Chế',
            icon: 'fa-recycle',
            color: 'var(--color-recycle)',
            desc: 'Bao gồm giấy, bìa carton, nhựa, thủy tinh và kim loại sạch.',
            rec: 'Hãy rửa sạch (nếu cần) và bỏ vào thùng rác Tái Chế (thường có màu Trắng/Xanh lá).'
        },
        'organic': {
            name: 'Rác Hữu Cơ',
            icon: 'fa-leaf',
            color: 'var(--color-organic)',
            desc: 'Thức ăn thừa, vỏ trái cây, bã trà, cà phê và các loại lá cây.',
            rec: 'Bỏ vào thùng rác Hữu Cơ (thường có màu Xanh lá cây) để ủ thành phân bón sinh học.'
        },
        'hazardous': {
            name: 'Rác Độc Hại',
            icon: 'fa-skull-crossbones',
            color: 'var(--color-hazardous)',
            desc: 'Pin, bóng đèn huỳnh quang, hóa chất, vỏ chai thuốc bảo vệ thực vật.',
            rec: 'TUYỆT ĐỐI KHÔNG vứt chung với rác thường. Hãy bỏ vào thùng rác Độc Hại (thường màu Đen/Đỏ) hoặc mang đến điểm thu gom.'
        },
        'solid': {
            name: 'Rác Vô Cơ Khác',
            icon: 'fa-trash-can',
            color: 'var(--color-solid)',
            desc: 'Túi nilon bẩn, hộp xốp, tã giấy, sành sứ vỡ.',
            rec: 'Bỏ vào thùng rác Vô Cơ (thường có màu Vàng hoặc Cam) để đưa đi chôn lấp hoặc đốt.'
        }
    };

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

    fileInput.addEventListener('change', function () {
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

        // UI Loading State
        classifyBtn.disabled = true;
        btnText.textContent = 'Đang Phân Tích...';
        btnIcon.classList.add('hidden');
        spinner.classList.remove('hidden');
        resultSection.classList.add('hidden');

        // Gửi biến `currentFile` lên Backend Python
        const formData = new FormData();
        formData.append('image', currentFile);

        fetch('/api/classify', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    const info = categoryInfo[data.category_id];

                    // Cập nhật giao diện
                    confidenceBadge.textContent = `${data.confidence}%`;
                    confidenceBadge.style.color = info.color;
                    confidenceBadge.style.borderColor = info.color;
                    confidenceBadge.style.background = `${info.color}20`; // 20 is alpha in hex

                    resultIconWrapper.style.background = `${info.color}20`;
                    resultIconWrapper.style.color = info.color;
                    resultIcon.className = `fa-solid ${info.icon}`;

                    resultTitle.textContent = info.name;
                    resultTitle.style.color = info.color;

                    resultDesc.textContent = info.desc;
                    recommendationText.textContent = info.rec;
                    actionRecommendation.style.borderLeftColor = info.color;

                    resultSection.classList.remove('hidden');

                    if (window.innerWidth < 600) {
                        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                } else {
                    alert('Lỗi từ Server: ' + data.error);
                }
            })
            .catch(err => {
                console.error(err);
                alert('Không thể kết nối đến máy chủ. Hãy đảm bảo file app.py đang chạy.');
            })
            .finally(() => {
                // Reset Button State
                classifyBtn.disabled = false;
                btnText.textContent = 'Phân Loại Lại';
                btnIcon.classList.remove('hidden');
                spinner.classList.add('hidden');
            });
    });
});
