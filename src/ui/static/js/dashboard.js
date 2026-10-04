let isPaused = false;
let ws = null;

document.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
    initControls();
    initFileUpload();
});

function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("Telemetry WebSocket connected.");
        document.getElementById("connection-status").textContent = "CONNECTED (LIVE)";
    };

    ws.onmessage = (event) => {
        if (isPaused) return;
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    ws.onclose = () => {
        console.log("WebSocket disconnected. Reconnecting in 2s...");
        document.getElementById("connection-status").textContent = "RECONNECTING...";
        setTimeout(initWebSocket, 2000);
    };
}

function updateDashboard(payload) {
    // 1. Update Video Stream Canvas
    if (payload.frame_base64) {
        const streamImg = document.getElementById("stream-canvas");
        streamImg.src = `data:image/jpeg;base64,${payload.frame_base64}`;
    }

    // 2. Update Overlay badges
    if (payload.latency_ms !== undefined) {
        document.getElementById("overlay-latency").textContent = `${payload.latency_ms} ms`;
        document.getElementById("stat-latency").textContent = `${payload.latency_ms} ms`;
    }
    if (payload.fps !== undefined) {
        document.getElementById("overlay-fps").textContent = `${payload.fps} FPS`;
    }

    const telem = payload.telemetry || {};

    // 3. Update Stat KPI Boxes
    if (telem.instant_throughput_tons_hr !== undefined) {
        document.getElementById("stat-throughput").textContent = `${telem.instant_throughput_tons_hr} t/h`;
        document.getElementById("stat-throughput-kg").textContent = `${telem.instant_throughput_kg_hr} kg/h`;
    }
    if (telem.total_items_processed !== undefined) {
        document.getElementById("stat-total-items").textContent = telem.total_items_processed;
        document.getElementById("stat-annex-vii").textContent = `${telem.annex_vii_mandatory_extractions} Annex VII`;
    }
    if (telem.gross_revenue_eur !== undefined) {
        document.getElementById("stat-gross-revenue").textContent = `€${telem.gross_revenue_eur.toFixed(2)}`;
    }
    if (telem.net_profit_eur !== undefined) {
        const netElem = document.getElementById("stat-net-profit");
        netElem.textContent = `€${telem.net_profit_eur.toFixed(2)}`;
        netElem.className = `stat-value ${telem.net_profit_eur >= 0 ? 'positive' : 'danger'}`;
        document.getElementById("stat-net-rate").textContent = `€${telem.net_rate_eur_per_hr.toFixed(2)}/hr`;
    }
    if (telem.energy_consumed_kwh !== undefined) {
        document.getElementById("stat-energy").textContent = `${telem.energy_consumed_kwh.toFixed(3)} kWh`;
        document.getElementById("stat-energy-cost").textContent = `Cost: €${telem.energy_cost_eur.toFixed(2)}`;
    }

    // 4. Update Dynamic AI Perception & Accuracy Benchmark Metrics
    const acc = payload.accuracy_metrics || {};
    if (acc.overall_pixel_accuracy !== undefined) {
        const elOa = document.getElementById("m-oa");
        if (elOa) elOa.textContent = `${(acc.overall_pixel_accuracy * 100).toFixed(2)}%`;
    }
    if (acc.per_class_metrics && acc.per_class_metrics["BFR Polymers"]) {
        const pMet = acc.per_class_metrics["BFR Polymers"];
        const elRec = document.getElementById("m-poly-rec");
        const elIou = document.getElementById("m-poly-iou");
        const elF1 = document.getElementById("m-poly-f1");
        if (elRec) elRec.textContent = `${(pMet.recall * 100).toFixed(2)}%`;
        if (elIou) elIou.textContent = `${(pMet.iou * 100).toFixed(2)}%`;
        if (elF1) elF1.textContent = `${(pMet.f1_dice * 100).toFixed(2)}%`;
    }

    // 4. Update Inspection Stream & Recovery Log Table
    const tbody = document.getElementById("inspection-tbody");
    const itemsToAdd = (payload.newly_sorted && payload.newly_sorted.length > 0)
        ? payload.newly_sorted
        : (tbody.children.length === 0 && payload.sorted_history && payload.sorted_history.length > 0 ? payload.sorted_history : []);

    if (itemsToAdd.length > 0) {
        itemsToAdd.forEach(item => {
            // Check if item track_id already rendered to avoid duplicate rows
            if (document.getElementById(`row-track-${item.track_id}`)) return;

            const tr = document.createElement("tr");
            tr.id = `row-track-${item.track_id}`;
            const tagClass = getTagClass(item.class_id);
            tr.innerHTML = `
                <td><strong>#${item.track_id}</strong></td>
                <td><span class="tag ${tagClass}">${item.class_name}</span></td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${item.area_cm2.toFixed(1)} cm²</td>
                <td>${item.estimated_mass_g.toFixed(1)} g</td>
                <td><code>${item.target_bin}</code></td>
                <td><strong>€${item.commodity_value_eur.toFixed(2)}</strong></td>
                <td><span class="tag ${item.is_hazardous ? 'tag-battery' : 'tag-pcb'}">${item.epr_compliance_status}</span></td>
            `;
            tbody.insertBefore(tr, tbody.firstChild);
            // Limit table to latest 20 items
            while (tbody.children.length > 20) {
                tbody.removeChild(tbody.lastChild);
            }
        });
    }
}

function getTagClass(cid) {
    switch (cid) {
        case 0: return "tag-pcb";
        case 1: return "tag-ic";
        case 2: return "tag-battery";
        case 3: return "tag-bfr";
        case 4: return "tag-metal";
        default: return "tag-metal";
    }
}

function initControls() {
    const pauseBtn = document.getElementById("btn-pause");
    pauseBtn.addEventListener("click", () => {
        isPaused = !isPaused;
        pauseBtn.textContent = isPaused ? "Resume Stream" : "Pause Stream";
    });

    const speedSlider = document.getElementById("slider-speed");
    const speedVal = document.getElementById("speed-val");
    speedSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        speedVal.textContent = `${val.toFixed(1)} m/s`;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: "set_speed", speed_mps: val }));
        }
    });

    const confSlider = document.getElementById("slider-conf");
    const confVal = document.getElementById("conf-val");
    confSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        confVal.textContent = `${Math.round(val * 100)}%`;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: "set_confidence", conf_threshold: val }));
        }
    });
}

function initFileUpload() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#00f2fe";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
        if (e.dataTransfer.files.length > 0) {
            uploadMedia(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadMedia(e.target.files[0]);
        }
    });
}

async function uploadMedia(file) {
    const statusBox = document.getElementById("upload-status");
    statusBox.textContent = `Processing ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        const res = await fetchJson(resp);
        if (res.success) {
            statusBox.textContent = `Completed: ${file.name} (${res.num_detections} items detected)`;
            if (res.annotated_base64) {
                document.getElementById("stream-canvas").src = `data:image/jpeg;base64,${res.annotated_base64}`;
            }
            if (res.telemetry) {
                updateDashboard({ telemetry: res.telemetry, newly_sorted: res.sorted_items });
            }
        } else {
            statusBox.textContent = `Error: ${res.error || 'Failed to process'}`;
        }
    } catch (err) {
        statusBox.textContent = `Upload error: ${err.message}`;
    }
}

async function fetchJson(resp) {
    return await resp.json();
}
