// static/js/modal.js
export function logUploadedFiles(scope = document) {
    for (let i = 1; i <= 8; i++) {
        const input = scope.querySelector(`#upload_${i}`);
        if (!input) {
            console.log(`⚠️ 找不到 upload_${i}`);
        } 
        else if (!input.files)
        {
            console.log(`⚠️ 找不到 upload_${i}.files`);
        }
        else
        {
          if (input.files.length === 0) {
              console.log(`❌ upload_${i} 沒有檔案`);
          } else {
              console.log(`✅ upload_${i} 有檔案：${input.files[0].name}`);
          }
        }
    }
}

export function bindImageResultPreview(parts, scope = document) {
    console.log("bindImageUploadPreview() called with scope:", scope);

    parts.forEach(([code, label]) => {
        const uploadInput = scope.querySelector(`#upload_${code}`);
    });
}

export function bindImageUploadPreview(parts, scope = document) {
    console.log("bindImageUploadPreview() called with scope:", scope);

    parts.forEach(([code, label]) => {
        const selectBtn = scope.querySelector(`#SelectBtn_${code}`);
        const uploadInput = scope.querySelector(`#upload_${code}`);
        const uploadInput2 = scope.querySelector(`#upload2_${code}`);
        const previewImg = scope.querySelector(`#preview_${code}`);

        uploadInput.addEventListener("change", (e) => {
          const file = e.target.files[0];
          if (file) {
              console.log(`選取檔案: ${file.name}`);
          } else {
              console.log("沒有選擇檔案");
          }
        });

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

                    showModal("請選擇要上傳圖片/實際拍攝", "modal-select", () => {
                        uploadInput.click();
                        console.log("使用者點擊 上傳圖片");
                    }, () => {
                        console.log("使用者點擊 實際拍攝");
                        const photo_modal = document.getElementById("PhotoModal");

                        if (photo_modal) {

                            const modal = bootstrap.Modal.getInstance(photo_modal); // 取得已存在的 modal 實例
                            if (modal) {
                                modal.show();
                            }
                        }

                        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
                            document.getElementById("video").srcObject = stream;
                        });

                        document.getElementById("capture").addEventListener("click", () => {
                            let video = document.getElementById("video");
                            let canvas = document.getElementById("canvas");
                            let ctx = canvas.getContext("2d");

                            // 把攝影機影像畫到 canvas
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                            // 轉成 Base64 傳給後端
                            let dataURL = canvas.toDataURL("image/jpeg");

                            fetch("/upload", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ image: dataURL })
                            })
                            .then(res => res.blob())
                            .then(blob => {
                                document.getElementById("result_img").src = URL.createObjectURL(blob);
                                document.getElementById("upload_img").value = `preview_${code}`;

                                if (previewImg) {
                                    console.log("previewImg.src正常運作 (previewImg.src=", previewImg.src, ")");
                                    previewImg.src = URL.createObjectURL(blob);
                                }
                            });
                        });
                    });
                });
                selectBtn.dataset.bound = "true";
            }

            if (!uploadInput.dataset.bound) {
                uploadInput.addEventListener("change", (e) => {
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function (e) {
                            if (previewImg) {
                                console.log("previewImg.src正常運作 (previewImg.src=", previewImg.src, ")");
                                previewImg.src = e.target.result;
                            }
                        };
                        reader.readAsDataURL(file); // base64 給 img
                        selectBtn.innerHTML = "已上傳";
                    }
                });
                uploadInput.dataset.bound = "true";
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
          <p id="modal-message" style="font-size:16px; margin-bottom:1.2rem;"></p>
          <div style="display:flex; justify-content:center; gap:1rem;">
            <button id="modal-ok-select" style="padding:0.6rem 1.2rem; background:#409EFF; color:white; border:none; border-radius:6px; cursor:pointer;">` + leftmodal_msg + `</button>
            <button id="modal-cancel-select" style="padding:0.6rem 1.2rem; background:#e0e0e0; color:#333; border:none; border-radius:6px; cursor:pointer;" data-bs-toggle="modal" data-bs-target="#PhotoModal">` + rightmodal_msg + `</button>
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

export function showModal(message, modal_name="modal", onOk = null, onCancel = null) {
    const modal = document.getElementById(modal_name);
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