// static/js/modal.js
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

export function loadModal(model_container_name) {
  const container = document.getElementById(model_container_name);
  container.innerHTML = `
    <div id="modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; 
        background:rgba(0,0,0,0.4); z-index:10000; display:none; align-items:center; justify-content:center;">
      <div style="background:white; padding:20px 30px; border-radius:10px; min-width:280px; max-width:400px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
        <p id="modal-message" style="font-size:16px; margin-bottom:1.2rem;"></p>
        <div style="display:flex; justify-content:center; gap:1rem;">
          <button id="modal-ok" style="padding:0.6rem 1.2rem; background:#409EFF; color:white; border:none; border-radius:6px; cursor:pointer;">OK</button>
          <button id="modal-cancel" style="padding:0.6rem 1.2rem; background:#e0e0e0; color:#333; border:none; border-radius:6px; cursor:pointer;">Cancel</button>
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

export function loadingModal(okCallback = null, cancelCallback = null) {
  const container = document.getElementById("modal-container-loading");
  if (!container) return;

  container.innerHTML = `
    <div class="modal fade" id="loadingModal" tabindex="-1" aria-hidden="true" data-bs-backdrop="static" data-bs-keyboard="false">
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content text-center">
          <div class="modal-body">
            <div class="spinner-border text-primary mb-3" role="status" style="width: 3rem; height: 3rem;">
              <span class="visually-hidden">Loading...</span>
            </div>
            <div>模型推論中，請稍候...</div>
            <p class="mt-3 mb-0" id="progressText-inference"></p>
            <div style="display:flex; justify-content:center; gap:1rem;">
              <button id="loading-cancel-btn" class="btn btn-secondary">Cancel</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const modal = document.getElementById("loadingModal");
  const modalInstance = new bootstrap.Modal(modal);
  modalInstance.show();

  const okBtn = document.getElementById("loading-ok-btn");
  const cancelBtn = document.getElementById("loading-cancel-btn");

  if (cancelBtn) {
    cancelBtn.onclick = () => {
      if (cancelCallback) cancelCallback();

      fetch("/cancel_inference", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          console.log("後端回應:", data);
          showModal("中止推論", () => console.log("推論已中止"));
          modalInstance.hide();
        });
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