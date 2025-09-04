// static/js/modal.js

const add_form = document.getElementById("add_form");
const edit_form = document.getElementById("edit_form");

export function logUploadedFiles(scope = document) {
    for (let i = 1; i <= 8; i++) {
        const input = scope.querySelector(`#upload_${i}`);
        if (!input) {
            console.log(`⚠️ 找不到 upload_${i}`);
        } 
        else if (!input.files) {
            console.log(`⚠️ 找不到 upload_${i}.files`);
        }
        else {
            if (input.files.length === 0) {
                console.log(`❌ upload_${i} 沒有檔案`);
            } else {
                console.log(`✅ upload_${i} 有檔案：${input.files[0].name}`);
            }
        }
    }
}

export function bindImageResultPreview(parts, scope = document) {
    console.log("bindImageResultPreview() called with scope:", scope);
    parts.forEach(([code, label]) => {
        const uploadInput = scope.querySelector(`#upload_${code}`);
    });
}

function showPhotoModal(code, scope_name) {
    const photo_modal = document.getElementById("PhotoModal");

    photo_modal.dataset.currentCode = code;
    photo_modal.dataset.prev_modalname = scope_name;

    // ✅ 把 code 塞進 captureBtn，避免 undefined
    const captureBtn = photo_modal.querySelector("#captureBtn");
    if (captureBtn) {
        captureBtn.dataset.code = code;
    }

    const modal = bootstrap.Modal.getOrCreateInstance(photo_modal);
    modal.show();

    const video = document.getElementById("video");
    const result_img = document.getElementById("result_img");

    video.style.display = "flex";
    result_img.src = "";

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; });
}

export function bindImageUploadPreview(parts, scope = document, scope_name = null) {
    const photo_modal = document.getElementById("PhotoModal");
    const video = photo_modal.querySelector("#video");
    const canvas = photo_modal.querySelector("#canvas");
    const ctx = canvas.getContext("2d"); 

    let scale = 1.0;
    let originX = 0;
    let originY = 0;
    let isDragging = false;
    let startX, startY;
    let animationId = null;

    function fresh() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(originX, originY);
        ctx.scale(2, 1);
        // ✅ 使用 video 原始大小，避免覆蓋 zoom/pan
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        ctx.restore();
        animationId = requestAnimationFrame(fresh);
    }

    function startPreview() {
        if (!animationId) {
            // ✅ 依 video 實際大小設定 canvas
            canvas.width = video.videoWidth || canvas.width;
            canvas.height = video.videoHeight || canvas.height;
            animationId = requestAnimationFrame(fresh);
        }
    }

    function stopPreview() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }

    // === Zoom / Pan ===
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        const zoomFactor = 1.1;
        const mouseX = (e.offsetX - originX) / scale;
        const mouseY = (e.offsetY - originY) / scale;

        if (e.deltaY < 0) {
            scale *= zoomFactor;
        } else {
            scale /= zoomFactor;
        }

        originX = e.offsetX - mouseX * scale;
        originY = e.offsetY - mouseY * scale;
    });

    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        startX = e.clientX - originX;
        startY = e.clientY - originY;
    });

    canvas.addEventListener("mousemove", (e) => {
        if (isDragging) {
            originX = e.clientX - startX;
            originY = e.clientY - startY;
        }
    });

    canvas.addEventListener("mouseup", () => isDragging = false);
    canvas.addEventListener("mouseleave", () => isDragging = false);

    // === Modal 開啟 / 關閉控制 ===
    photo_modal.addEventListener("shown.bs.modal", () => {
        console.log("PhotoModal opened");
        startPreview();
    });

    photo_modal.addEventListener("hidden.bs.modal", () => {
        console.log("PhotoModal closed");
        stopPreview();
    });

    parts.forEach(([code, label]) => {
        const selectBtn = scope.querySelector(`#SelectBtn_${code}`);
        const uploadInput = scope.querySelector(`#upload_${code}`);
        const uploadInput2 = scope.querySelector(`#upload2_${code}`);
        const previewImg = scope.querySelector(`#preview_${code}`);
        const captureBtn = photo_modal.querySelector("#captureBtn");

        if (captureBtn) {
            captureBtn.addEventListener("click", (e) => {
                e.preventDefault();
                const code = e.target.dataset.code;  // ✅ 這裡已確保有值

                const result_img = document.getElementById("result_img");

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    const file = new File([blob], `capture_${code}.png`, { type: "image/png" });
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);

                    const input = document.createElement("input");
                    input.type = "file";
                    input.name = `file_${code}`;
                    input.files = dataTransfer.files;
                    input.hidden = true;

                    document.getElementById("upload_form").appendChild(input);
                }, "image/png");

                result_img.src = canvas.toDataURL("image/png");
            });
        }

        if (uploadInput) {
            uploadInput.addEventListener("change", async function(e) {
                const file = e.target.files[0];
                if (file) {
                    console.log(`選取檔案: ${file.name}`);
                    const formData = new FormData();
                    formData.append("file", file);

                    const res = await fetch("/upload_file", {
                        method: "POST",
                        body: formData
                    });
                    const data = await res.json();

                    previewImg.src = data.url; // 後端回傳的 URL
                    selectBtn.innerHTML = "已上傳";
                } else {
                    console.log("沒有選擇檔案");
                }
            });
        }

        if (previewImg && selectBtn) {
            const url = new URL(previewImg.src);
            if (!url.pathname.includes("/guide/")) {
                selectBtn.innerHTML = "已上傳";
            }
        }

        if (selectBtn && (uploadInput || uploadInput2)) {
            if (!selectBtn.dataset.bound) {
                selectBtn.addEventListener("click", () => {
                    console.log("uploadInput觸發");
                    showSelectModal("請選擇要上傳圖片/實際拍攝", () => {
                        uploadInput.click();
                        console.log("使用者點擊 上傳圖片");
                    }, () => {
                        console.log("使用者點擊 實際拍攝");
                        showPhotoModal(code, scope_name);
                    });
                });
                selectBtn.dataset.bound = "true";
            }
        }
    });
}

export function showLoadingModal() {
    const modalEl = document.getElementById('loadingModal');
    if (!modalEl) {
        console.error("找不到 #loadingModal，請確認 loadModal() 有正確呼叫");
        return null;
    }
    const modal = new bootstrap.Modal(modalEl, {
        backdrop: 'static',
        keyboard: false
    });
    modal.show();
    return modal;
}

export function showLoadingModal2() {
    const modalEl = document.getElementById('loadingModal2');
    if (!modalEl) {
        console.error("找不到 #loadingModal2，請確認 loadModal2() 有正確呼叫");
        return null;
    }
    const modal = new bootstrap.Modal(modalEl, {
        backdrop: 'static',
        keyboard: false
    });
    modal.show();
    return modal;
}

export function hideLoadingModal() {
    const modalEl = document.getElementById('loadingModal');
    if (!modalEl) return;
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) modal.hide();
}

export function hideLoadingModal2() {
    const modalEl = document.getElementById('loadingModal2');
    if (!modalEl) return;
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) modal.hide();
}

export function SelectModal(model_container_name, leftmodal_msg="上傳影像", rightmodal_msg="拍照") {
    const container = document.getElementById(model_container_name);
    container.innerHTML = `
      <div id="modal-select" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; 
          background:rgba(0,0,0,0.4); z-index:10000; display:none; align-items:center; justify-content:center;">
        <div style="background:white; padding:20px 30px; border-radius:10px; min-width:280px; max-width:400px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
          <p id="modal-message" style="font-size:16px; margin-bottom:1.2rem;"> 請選擇要上傳影像還是拍照? </p>
          <div style="display:flex; justify-content:center; gap:1rem;">
            <button id="modal-ok-select">` + leftmodal_msg + `</button>
            <button id="modal-cancel-select" data-bs-toggle="modal" data-bs-target="#PhotoModal">` + rightmodal_msg + `</button>
          </div>
        </div>
      </div>
    `;

    const modal = document.getElementById("modal-select");
    const model_load = new bootstrap.Modal(modal);
    const okBtn = document.getElementById("modal-ok-select");
    const cancelBtn = document.getElementById("modal-cancel-select");

    okBtn.onclick = () => {
        if (modal.okCallback) modal.okCallback();
        model_load.hide();
        modal.style.display = "none";
    };

    cancelBtn.onclick = () => {
        if (modal.cancelCallback) modal.cancelCallback();
        model_load.hide();
        modal.style.display = "none";
    };
}

export function loadModal(model_container_name, leftmodal_msg="OK", rightmodal_msg="Cancel") {
    const container = document.getElementById(model_container_name);
    container.innerHTML = `
      <div id="modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; 
          background:rgba(0,0,0,0.4); z-index:10000; display:none; align-items:center; justify-content:center;">
        <div style="background:white; padding:20px 30px; border-radius:10px; min-width:280px; max-width:400px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
          <p id="modal-message" style="font-size:16px; margin-bottom:1.2rem;"></p>
          <div style="display:flex; justify-content:center; gap:1rem;">
            <button id="modal-ok" style="padding:0.6rem 1.2rem; background:#409EFF; color:white; border:none; border-radius:6px; cursor:pointer;">` + leftmodal_msg + `</button>
            <button id="modal-cancel" style="padding:0.6rem 1.2rem; background:#e0e0e0; color:#333; border:none; border-radius:6px; cursor:pointer;">` + rightmodal_msg + `</button>
          </div>
        </div>
      </div>
    `;

    const modal = document.getElementById("modal");
    const model_load = new bootstrap.Modal(modal);
    const okBtn = document.getElementById("modal-ok");
    const cancelBtn = document.getElementById("modal-cancel");

    okBtn.onclick = () => {
        if (modal.okCallback) modal.okCallback();
        model_load.hide();
        modal.style.display = "none";
    };

    cancelBtn.onclick = () => {
        if (modal.cancelCallback) modal.cancelCallback();
        model_load.hide();
        modal.style.display = "none";
    };
}

export function loadingModal(message=null, hideCallback = null, cancelCallback = null, patient_id = null, inference = true) {
    const container = document.getElementById("modal-container-loading");
    if (!container) return;

    container.innerHTML = `
      <div class="modal fade" id="loadingModal" tabindex="-1" aria-labelledby="loadingLabel" aria-hidden="true" data-bs-backdrop="static">
        <div class="modal-dialog modal-dialog-centered">
          <div class="modal-content text-center p-4">
            <h5 class="modal-title" id="loadingLabel">` + message + `</h5>
            <div class="progress mt-3 w-100">
                <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%">0%</div>
            </div>
            <p class="mt-3 mb-0" id="progressText"></p>
            <div style="display:flex; justify-content:center; gap:1rem;">
                <button id="loading-hide-btn" class="btn btn-secondary">Hide</button>
                <button id="loading-cancel-btn" class="btn btn-secondary">Cancel</button>
            </div>
          </div>
        </div>
      </div>
    `;

    const modal = document.getElementById("loadingModal");
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();

    const hideBtn = document.getElementById("loading-hide-btn");
    const cancelBtn = document.getElementById("loading-cancel-btn");

    if (hideBtn) {
        hideBtn.onclick = () => {
            if (hideCallback) hideCallback();
            modalInstance.hide();
        }
    }

    if (cancelBtn) {
        cancelBtn.onclick = () => {
            if (cancelCallback) cancelCallback();
            
            if (inference === true) {
              if (patient_id) {
                fetch(`/record/cancel_inference/${patient_id}`, { method: "POST" })
                    .then(res => res.json())
                    .then(data => {
                      console.log("後端回應:", data);
                      showModal("中止推論", () => console.log("推論已中止"));
                      modalInstance.hide();
                    });
              }
              else {
                fetch("/cancel_inference", { method: "POST" })
                    .then(res => res.json())
                    .then(data => {
                      console.log("後端回應:", data);
                      showModal("中止推論", () => console.log("推論已中止"));
                      modalInstance.hide();
                    });
              }
            } else {
               modalInstance.hide();
            } 
        };
    }
}

export function loadingModal2(okCallback = null, cancelCallback = null) {
    const container = document.getElementById("modal-container-loading-2");
    if (!container) return;

    container.innerHTML = `
      <div class="modal fade" id="loadingModal2" tabindex="-1" aria-labelledby="loadingLabel" aria-hidden="true" data-bs-backdrop="static">
        <div class="modal-dialog modal-dialog-centered">
          <div class="modal-content text-center p-4">
            <h5 class="modal-title" id="loadingLabel">模型訓練中...</h5>
            <div class="progress mt-3 w-100">
                <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%">0%</div>
            </div>
            <p class="mt-3 mb-0" id="progressText"></p>
            <div style="display:flex; justify-content:center; gap:1rem;">
                <button id="modal-ok-2" class="btn btn-primary"> Hide training phase </button>
                <button id="modal-cancel-2" class="btn btn-secondary">Cancel</button>
            </div>
          </div>
        </div>
      </div>
    `;

    const modal = document.getElementById("loadingModal2");
    const okBtn = document.getElementById("modal-ok-2");
    const cancelBtn = document.getElementById("modal-cancel-2");

    if (okBtn) {
        okBtn.onclick = () => {
            if (okCallback) okCallback();
            bootstrap.Modal.getInstance(modal)?.hide();
            modal.style.display = "none";
        };
    }

    if (cancelBtn) {
        cancelBtn.onclick = () => {
            if (cancelCallback) cancelCallback();

            fetch("/cancel_training", { method: "POST" })
                .then(res => res.json())
                .then(data => {
                  console.log("後端回應:", data);
                  showModal("中止訓練", () => console.log("訓練已中止"));
                  bootstrap.Modal.getInstance(modal)?.hide();
                  modal.style.display = "none";
                });
        };
    }

    new bootstrap.Modal(modal).show();
}

export function showModal(message, onOk = null, onCancel = null) {
    const modal = document.getElementById("modal");
    document.getElementById("modal-message").innerText = message;
    modal.okCallback = onOk;
    modal.cancelCallback = onCancel;
    modal.style.display = "flex";
}

export function showSelectModal(message, onOk = null, onCancel = null) {
    const modal = document.getElementById("modal-select");
    document.getElementById("modal-message").innerText = message;
    modal.okCallback = onOk;
    modal.cancelCallback = onCancel;
    modal.style.display = "flex";
}

export function loadModal_custom() {
    const container = document.getElementById("");

    container.innerHTML = `
      <div class="modal fade" id="customModal" tabindex="-1" aria-labelledby="customModalLabel" aria-hidden="true">
        <div class="modal-dialog">
          <div class="modal-content">
            <div class="modal-header">
              <h5 class="modal-title" id="customModalLabel">提示訊息</h5>
              <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="關閉"></button>
            </div>
            <div class="modal-body" id="customModalBody">這裡是訊息內容</div>
            <div class="modal-footer">
              <button type="button" class="btn btn-secondary" id="customCancelBtn" data-bs-dismiss="modal">取消</button>
              <button type="button" class="btn btn-primary" id="customOkBtn" data-bs-dismiss="modal">OK</button>
            </div>
          </div>
        </div>
      </div>
    `;
}

export function showModal_custom(message, okCallback = null, cancelCallback = null) {
    const modalBody = document.getElementById("customModalBody");
    modalBody.innerText = message;

    const modal = new bootstrap.Modal(document.getElementById("customModal"));
    modal.show();

    // 清除上一次綁定
    document.getElementById("customOkBtn").onclick = null;
    document.getElementById("customCancelBtn").onclick = null;

    if (okCallback) {
        document.getElementById("customOkBtn").onclick = okCallback;
    }

    if (cancelCallback) {
        document.getElementById("customCancelBtn").onclick = cancelCallback;
    }
}