/**
 * SansKrypt Main Client Script
 * Handles drag-and-drop uploads, instant previews, copy buttons, and interactive elements.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. File Upload Drag-and-Drop & Preview Handling
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("manuscript_image");
    const previewContainer = document.getElementById("image-preview-container");
    const previewImage = document.getElementById("preview-image");
    const uploadPrompt = document.getElementById("upload-prompt");
    const uploadForm = document.getElementById("upload-form");
    const submitBtn = document.getElementById("submit-btn");
    const processingSpinner = document.getElementById("processing-spinner");

    if (dropzone && fileInput) {
        // Click dropzone to open file dialog
        dropzone.addEventListener("click", () => fileInput.click());

        // Dragover and dragleave effects
        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        // Handle file drop
        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                showImagePreview(files[0]);
            }
        });

        // Handle file input change
        fileInput.addEventListener("change", (e) => {
            if (fileInput.files && fileInput.files.length > 0) {
                showImagePreview(fileInput.files[0]);
            }
        });
    }

    function showImagePreview(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (.png, .jpg, .jpeg, .webp, .tiff, .bmp).");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImage && previewContainer && uploadPrompt) {
                previewImage.src = e.target.result;
                previewContainer.classList.remove("d-none");
                uploadPrompt.classList.add("d-none");
            }
        };
        reader.readAsDataURL(file);
    }

    // 2. Form Submission Progress Indicator
    if (uploadForm && submitBtn && processingSpinner) {
        uploadForm.addEventListener("submit", () => {
            submitBtn.disabled = true;
            processingSpinner.classList.remove("d-none");
            processingSpinner.style.display = "block";
            submitBtn.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Digitizing & Translating Manuscript...
            `;
        });
    }

    // 3. Copy to Clipboard Functionality
    const copyButtons = document.querySelectorAll(".btn-copy");
    copyButtons.forEach(button => {
        button.addEventListener("click", () => {
            const targetId = button.getAttribute("data-target");
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                const textToCopy = targetElement.innerText || targetElement.textContent;
                navigator.clipboard.writeText(textToCopy.trim()).then(() => {
                    const originalHtml = button.innerHTML;
                    button.innerHTML = '<i class="bi bi-check-lg me-1"></i> Copied!';
                    button.classList.add("btn-success");
                    setTimeout(() => {
                        button.innerHTML = originalHtml;
                        button.classList.remove("btn-success");
                    }, 2000);
                }).catch(err => {
                    console.error("Failed to copy text: ", err);
                });
            }
        });
    });
});
