// App initialization and UI event hooks
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initDatasetSelector();
    initAcademicTables();
    initSensitivitySweeper();
    
    // Load initial dataset
    loadDataset("SKAB");
});

let currentChart = null;

// Tab switcher logic
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    const sections = document.querySelectorAll(".tab-section");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");
    
    const titles = {
        explorer: {
            title: "İnteraktif Zaman Serisi Anomali Gezgini",
            subtitle: "Modellerin anomali tahmin başarılarının ve anlık sinyal grafiklerinin karşılaştırmalı analizi"
        },
        explainability: {
            title: "Olasılıksal Açıklanabilirlik Konsolu",
            subtitle: "TimeSeriesAutomata modelinin durum geçiş olasılıklarını ve karar gerekçelerini inceleyin"
        },
        tables: {
            title: "Akademik Sonuç Tabloları",
            subtitle: "Makalede yer alan tüm performans, gürültü direnci, çalışma süresi ve önemlilik tabloları"
        },
        gallery: {
            title: "Görsel Rapor Galerisi",
            subtitle: "Birim testlerden ve analizlerden üretilen 12 akademik figürün yüksek çözünürlüklü kopyaları"
        },
        sensitivity: {
            title: "Parametre Hassasiyet Simülatörü",
            subtitle: "Pencere boyutu (w) ve Alfabe boyutunun (a) F1-skorları üzerindeki anlık simülasyonu"
        }
    };

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            
            navButtons.forEach(b => b.classList.remove("active"));
            sections.forEach(s => s.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(`tab-${tabId}`).classList.add("active");
            
            if (titles[tabId]) {
                pageTitle.textContent = titles[tabId].title;
                pageSubtitle.textContent = titles[tabId].subtitle;
            }
        });
    });
}

// Dataset select handler
function initDatasetSelector() {
    const selector = document.getElementById("dataset-select");
    selector.addEventListener("change", (e) => {
        loadDataset(e.target.value);
    });
}

// Load selected dataset variables and render charts
function loadDataset(datasetName) {
    // 1. Update metric cards dynamically from ROBUSTNESS_DATA
    const automataF1 = document.getElementById("automata-f1");
    const cnnF1 = document.getElementById("cnn-f1");
    const speedupVal = document.getElementById("speedup-val");
    
    const autoData = ROBUSTNESS_DATA.find(r => r.dataset === datasetName && r.model === "Automata");
    const cnnData = ROBUSTNESS_DATA.find(r => r.dataset === datasetName && r.model === "CNN");
    
    if (autoData && cnnData) {
        automataF1.textContent = autoData.orig_f1.toFixed(4);
        cnnF1.textContent = cnnData.orig_f1.toFixed(4);
    }
    
    if (BASELINE_DATA && BASELINE_DATA[datasetName]) {
        const autoTrainTime = BASELINE_DATA[datasetName]["Automata"]?.train_time || 0.01;
        const lstmTrainTime = BASELINE_DATA[datasetName]["LSTM"]?.train_time || 0.01;
        const speedup = Math.round(lstmTrainTime / autoTrainTime);
        speedupVal.textContent = `~${speedup}x Faster`;
    } else {
        speedupVal.textContent = "-";
    }
    
    // 2. Render Time-Series Chart
    renderChart(datasetName);
    
    // 3. Load default explanation
    loadLocalExplanation(datasetName, 0);
    
    // 4. Update tables content
    renderDynamicTables(datasetName);
    
    // 5. Update sensitivity sweep simulator gauge values
    updateSensitivityGauge();
}

// Draw interactive anomaly signal chart using Chart.js
function renderChart(datasetName) {
    const ctx = document.getElementById("timeSeriesChart").getContext("2d");
    const sampleData = datasetName === "SKAB" ? SKAB_SAMPLE : BATADAL_SAMPLE;
    
    const labels = sampleData.map(d => d.time.split(" ")[1] || d.time); // extract hour/min if datetime
    const values = sampleData.map(d => d.value);
    
    const truthAnoms = sampleData.map((d, i) => d.anomaly === 1 ? d.value : null);
    const automataAnoms = sampleData.map((d, i) => d.pred_automata === 1 ? d.value : null);
    const cnnAnoms = sampleData.map((d, i) => d.pred_cnn === 1 ? d.value : null);
    
    if (currentChart) {
        currentChart.destroy();
    }
    
    const data = {
        labels: labels,
        datasets: [
            {
                label: "Sensör Sinyali (PC1)",
                data: values,
                borderColor: "#64748b",
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.1
            },
            {
                label: "True Anomaly",
                data: truthAnoms,
                borderColor: "#f43f5e",
                backgroundColor: "#f43f5e",
                pointRadius: 6,
                showLine: false,
                pointStyle: "circle"
            },
            {
                label: "Automata Pred",
                data: automataAnoms,
                borderColor: "#6366f1",
                backgroundColor: "rgba(99, 102, 241, 0.2)",
                pointRadius: 8,
                pointStyle: "triangle",
                showLine: false
            },
            {
                label: "CNN Pred",
                data: cnnAnoms,
                borderColor: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.2)",
                pointRadius: 8,
                pointStyle: "rect",
                showLine: false
            }
        ]
    };
    
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        onClick: (event, elements) => {
            if (elements.length > 0) {
                const elementIndex = elements[0].index;
                const clickedItem = sampleData[elementIndex];
                
                // Switch to Explainability tab dynamically to showcase detail
                document.querySelector('[data-tab="explainability"]').click();
                loadLocalExplanation(datasetName, elementIndex);
            }
        },
        scales: {
            x: {
                grid: { color: "rgba(255, 255, 255, 0.03)" },
                ticks: { color: "#94a3b8" }
            },
            y: {
                grid: { color: "rgba(255, 255, 255, 0.03)" },
                ticks: { color: "#94a3b8" }
            }
        },
        plugins: {
            legend: { display: false }
        }
    };
    
    currentChart = new Chart(ctx, {
        type: "line",
        data: data,
        options: options
    });
    
    // Bind toggles checkboxes
    document.getElementById("toggle-truth").onchange = (e) => {
        currentChart.setDatasetVisibility(1, e.target.checked);
        currentChart.update();
    };
    document.getElementById("toggle-automata").onchange = (e) => {
        currentChart.setDatasetVisibility(2, e.target.checked);
        currentChart.update();
    };
    document.getElementById("toggle-cnn").onchange = (e) => {
        currentChart.setDatasetVisibility(3, e.target.checked);
        currentChart.update();
    };
}

// Load explainability details into Section 2 panel card
function loadLocalExplanation(datasetName, index) {
    const list = EXPLAIN_DATA[datasetName];
    if (!list || list.length === 0) return;
    
    const item = list[index % list.length];
    
    document.getElementById("explain-status").textContent = item.status;
    document.getElementById("explain-status").className = `badge ${item.status}`;
    
    document.getElementById("explain-step").textContent = item.time_step;
    document.getElementById("explain-state").textContent = item.state;
    document.getElementById("explain-pattern").textContent = item.pattern;
    
    if (item.status === "unseen") {
        document.getElementById("explain-mapping").textContent = `${item.mapped_to} (Edit Distance: ${item.distance})`;
    } else {
        document.getElementById("explain-mapping").textContent = "Gerekmiyor (Sözlükte var)";
    }
    
    document.getElementById("explain-prob").textContent = item.probability.toFixed(4);
    
    const decisionElem = document.getElementById("explain-decision");
    decisionElem.textContent = item.decision;
    decisionElem.className = `badge-decision ${item.decision}`;
    
    document.getElementById("explain-confidence").textContent = `${(item.confidence_score * 100).toFixed(2)}%`;
    document.getElementById("explain-reason").textContent = item.reason;
    
    // Populate transition probabilities listing
    const transList = document.getElementById("explain-transitions");
    transList.innerHTML = "";
    
    item.transitions.forEach(tr => {
        const trItem = document.createElement("div");
        trItem.className = "transition-item";
        trItem.innerHTML = `
            <div class="transition-path">
                <span>${tr.from}</span>
                <span class="arrow">➔</span>
                <span>${tr.to}</span>
            </div>
            <div class="transition-prob">${tr.probability.toFixed(4)}</div>
        `;
        transList.appendChild(trItem);
    });
}

// Handle tables tabs switching
function initAcademicTables() {
    const tabBtns = document.querySelectorAll(".table-tab-btn");
    const contents = document.querySelectorAll(".table-content");
    
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tableId = btn.getAttribute("data-table");
            
            tabBtns.forEach(b => b.classList.remove("active"));
            contents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(`table-${tableId}`).classList.add("active");
        });
    });
}

// Render Table 1, Table 2, Table 3, Table 4, Table 5, and Table 6 dynamically in HTML
function renderDynamicTables(datasetName) {
    // 0. Table 1: Model Performance and Stability (Means and Stds)
    const t1Body = document.getElementById("dyn-table-1-body");
    if (t1Body && BASELINE_DATA) {
        t1Body.innerHTML = "";
        const models = ["LSTM", "GRU", "CNN", "Automata"];
        
        for (const model of models) {
            const skabData = BASELINE_DATA["SKAB"]?.[model];
            const batData = BASELINE_DATA["BATADAL"]?.[model];
            
            const skabText = skabData ? `${skabData.f1_mean.toFixed(4)} ± ${skabData.f1_std.toFixed(4)}` : "-";
            const batText = batData ? `${batData.f1_mean.toFixed(4)} ± ${batData.f1_std.toFixed(4)}` : "-";
            
            const tr = document.createElement("tr");
            let modelLabel = `<strong>${model} Baseline</strong>`;
            if (model === "Automata") {
                modelLabel = `<strong class="text-indigo">TimeSeriesAutomata</strong>`;
            } else if (model === "CNN") {
                modelLabel = `<strong>1D-CNN Baseline</strong>`;
            }
            
            tr.innerHTML = `
                <td>${modelLabel}</td>
                <td>${skabText}</td>
                <td>${batText}</td>
            `;
            t1Body.appendChild(tr);
        }
    }

    // 0.5 Table 5: Runtime Comparisons
    const t5Body = document.getElementById("dyn-table-5-body");
    if (t5Body && BASELINE_DATA) {
        t5Body.innerHTML = "";
        const models = ["LSTM", "GRU", "CNN", "Automata"];
        
        for (const dname of ["SKAB", "BATADAL"]) {
            for (const model of models) {
                const data = BASELINE_DATA[dname]?.[model];
                if (data) {
                    const tr = document.createElement("tr");
                    let modelLabel = `<strong>${model}</strong>`;
                    if (model === "Automata") {
                        modelLabel = `<strong class="text-indigo">Automata</strong>`;
                    } else if (model === "CNN") {
                        modelLabel = `<strong>1D-CNN</strong>`;
                    }
                    
                    tr.innerHTML = `
                        <td>${modelLabel}</td>
                        <td>${dname}</td>
                        <td>${data.train_time.toFixed(4)}</td>
                        <td>${data.inference_time.toFixed(4)}</td>
                    `;
                    t5Body.appendChild(tr);
                }
            }
        }
    }

    // 1. Table 2: Robustness to Noise
    const t2 = document.getElementById("dyn-table-2");
    t2.innerHTML = `
        <thead>
            <tr>
                <th>Model</th>
                <th>Veri Seti</th>
                <th>Orijinal F1</th>
                <th>Gürültü F1 (0.05)</th>
                <th>Gürültü F1 (0.1)</th>
                <th>Gürültü F1 (0.15)</th>
                <th>Gürültü F1 (0.2)</th>
                <th>Gürültü F1 (0.25)</th>
            </tr>
        </thead>
        <tbody id="dyn-table-2-body"></tbody>
    `;
    const t2Body = document.getElementById("dyn-table-2-body");
    
    // Filter robustness data for SKAB and BATADAL
    const models = ["Automata", "LSTM", "GRU", "CNN"];
    for (const dname of ["SKAB", "BATADAL"]) {
        for (const model of models) {
            const sub = ROBUSTNESS_DATA.filter(r => r.dataset === dname && r.model === model);
            if (sub.length > 0) {
                const origF1 = sub[0].orig_f1.toFixed(4);
                const noiseF1s = [];
                for (const scale of [0.05, 0.1, 0.15, 0.2, 0.25]) {
                    const row = sub.find(r => Math.abs(r.noise_scale - scale) < 1e-4);
                    noiseF1s.push(row ? row.noisy_f1.toFixed(4) : "-");
                }
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${model}</strong></td>
                    <td>${dname}</td>
                    <td>${origF1}</td>
                    <td>${noiseF1s[0]}</td>
                    <td>${noiseF1s[1]}</td>
                    <td>${noiseF1s[2]}</td>
                    <td>${noiseF1s[3]}</td>
                    <td>${noiseF1s[4]}</td>
                `;
                t2Body.appendChild(tr);
            }
        }
    }

    // Helper trick since list append doesn't exist in vanilla JS
    function appendHelper(array, val) {
        array.push(val);
    }
    
    // 2. Table 3: Cross-Dataset comparison
    const t3 = document.getElementById("dyn-table-3");
    t3.innerHTML = "";
    
    for (const model of models) {
        const box = document.createElement("div");
        box.className = "cross-model-box";
        box.innerHTML = `
            <h5>Model: ${model}</h5>
            <table class="academic-table">
                <thead>
                    <tr>
                        <th>Train \\ Test</th>
                        <th>SKAB</th>
                        <th>BATADAL</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Train: SKAB</strong></td>
                        <td id="cross-${model}-skab-skab">-</td>
                        <td id="cross-${model}-skab-batadal">-</td>
                    </tr>
                    <tr>
                        <td><strong>Train: BATADAL</strong></td>
                        <td id="cross-${model}-batadal-skab">-</td>
                        <td id="cross-${model}-batadal-batadal">-</td>
                    </tr>
                </tbody>
            </table>
        `;
        t3.appendChild(box);
        
        // Find in-domain
        const subSkab = ROBUSTNESS_DATA.find(r => r.dataset === "SKAB" && r.model === model);
        const subBat = ROBUSTNESS_DATA.find(r => r.dataset === "BATADAL" && r.model === model);
        if (subSkab) document.getElementById(`cross-${model}-skab-skab`).textContent = subSkab.orig_f1.toFixed(4);
        if (subBat) document.getElementById(`cross-${model}-batadal-batadal`).textContent = subBat.orig_f1.toFixed(4);
        
        // Find cross-domain
        const crossSB = CROSS_DATASET_DATA.find(c => c.train_dataset === "SKAB" && c.test_dataset === "BATADAL" && c.model === model);
        const crossBS = CROSS_DATASET_DATA.find(c => c.train_dataset === "BATADAL" && c.test_dataset === "SKAB" && c.model === model);
        if (crossSB) document.getElementById(`cross-${model}-skab-batadal`).textContent = crossSB.f1.toFixed(4);
        if (crossBS) document.getElementById(`cross-${model}-batadal-skab`).textContent = crossBS.f1.toFixed(4);
    }
    
    // 3. Table 4: Sensitivity analysis
    const t4 = document.getElementById("dyn-table-4");
    t4.innerHTML = `
        <thead>
            <tr>
                <th>Veri Seti</th>
                <th>Parametre</th>
                <th>Değer = 3</th>
                <th>Değer = 4</th>
                <th>Değer = 5</th>
                <th>Değer = 6</th>
            </tr>
        </thead>
        <tbody id="dyn-table-4-body"></tbody>
    `;
    const t4Body = document.getElementById("dyn-table-4-body");
    for (const dname of ["SKAB", "BATADAL"]) {
        // window_size row
        const wSub = SENSITIVITY_DATA.filter(s => s.dataset === dname && s.parameter === "window_size");
        const wVals = [3, 4, 5, 6].map(v => {
            const r = wSub.find(s => s.value === v);
            return r ? r.f1.toFixed(4) : "-";
        });
        const wTr = document.createElement("tr");
        wTr.innerHTML = `
            <td><strong>${dname}</strong></td>
            <td>Pencere Boyutu (w)</td>
            <td>${wVals[0]}</td>
            <td>${wVals[1]}</td>
            <td>${wVals[2]}</td>
            <td>${wVals[3]}</td>
        `;
        t4Body.appendChild(wTr);
        
        // alphabet_size row
        const aSub = SENSITIVITY_DATA.filter(s => s.dataset === dname && s.parameter === "alphabet_size");
        const aVals = [3, 4, 5, 6].map(v => {
            const r = aSub.find(s => s.value === v);
            return r ? r.f1.toFixed(4) : "-";
        });
        const aTr = document.createElement("tr");
        aTr.innerHTML = `
            <td><strong>${dname}</strong></td>
            <td>Alfabe Boyutu (a)</td>
            <td>${aVals[0]}</td>
            <td>${aVals[1]}</td>
            <td>${aVals[2]}</td>
            <td>${aVals[3]}</td>
        `;
        t4Body.appendChild(aTr);
    }
    
    // 4. Table 6: Statistical tests
    const t6 = document.getElementById("dyn-table-6");
    t6.innerHTML = `
        <thead>
            <tr>
                <th>Veri Seti</th>
                <th>Karşılaştırma</th>
                <th>McNemar p-Değeri</th>
                <th>McNemar Sonuç</th>
                <th>Wilcoxon p-Değeri</th>
                <th>Wilcoxon Sonuç</th>
            </tr>
        </thead>
        <tbody id="dyn-table-6-body"></tbody>
    `;
    const t6Body = document.getElementById("dyn-table-6-body");
    for (const dname of ["SKAB", "BATADAL"]) {
        const dStats = STATISTICAL_DATA[dname] || {};
        const comps = dStats.comparisons || {};
        for (const base of ["LSTM", "GRU", "CNN"]) {
            const pair = comps[`Automata_vs_${base}`];
            if (pair) {
                const mc = pair.mcnemar || {};
                const wil = pair.wilcoxon || {};
                
                const mc_p = typeof mc.p_value === "number" ? mc.p_value.toFixed(4) : mc.p_value;
                const mc_sig = mc.significant ? "Anlamlı (H1)" : "Geçersiz (H0)";
                const wil_p = typeof wil.p_value === "number" ? wil.p_value.toFixed(4) : wil.p_value;
                const wil_sig = wil.significant ? "Anlamlı (H1)" : "Geçersiz (H0)";
                
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${dname}</strong></td>
                    <td>Automata vs ${base}</td>
                    <td>${mc_p}</td>
                    <td><span class="badge-decision ${mc.significant ? 'anomaly' : 'normal'}">${mc_sig}</span></td>
                    <td>${wil_p}</td>
                    <td><span class="badge-decision ${wil.significant ? 'anomaly' : 'normal'}">${wil_sig}</span></td>
                `;
                t6Body.appendChild(tr);
            }
        }
    }
}

// Binds range sliders and updates simulated F1 metric gauges
function initSensitivitySweeper() {
    const slideWindow = document.getElementById("slider-window");
    const slideAlphabet = document.getElementById("slider-alphabet");
    
    slideWindow.addEventListener("input", (e) => {
        document.getElementById("val-window").textContent = e.target.value;
        updateSensitivityGauge();
    });
    
    slideAlphabet.addEventListener("input", (e) => {
        document.getElementById("val-alphabet").textContent = e.target.value;
        updateSensitivityGauge();
    });
}

// Gauge and interpretation updates
function updateSensitivityGauge() {
    const dataset = document.getElementById("dataset-select").value;
    const w = parseInt(document.getElementById("slider-window").value);
    const a = parseInt(document.getElementById("slider-alphabet").value);
    
    // Find closest sensitivity values
    // Since we sweep window_size and alphabet_size separately:
    // We can prioritize window_size sweep or alphabet_size sweep
    const wRow = SENSITIVITY_DATA.find(s => s.dataset === dataset && s.parameter === "window_size" && s.value === w);
    const aRow = SENSITIVITY_DATA.find(s => s.dataset === dataset && s.parameter === "alphabet_size" && s.value === a);
    
    // Average or take window F1
    let simulatedF1 = 0;
    if (wRow && aRow) {
        simulatedF1 = (wRow.f1 + aRow.f1) / 2;
    } else if (wRow) {
        simulatedF1 = wRow.f1;
    } else if (aRow) {
        simulatedF1 = aRow.f1;
    } else {
        simulatedF1 = 0.5;
    }
    
    document.getElementById("sens-f1-val").textContent = simulatedF1.toFixed(4);
    
    // Update gauge arc length stroke-dashoffset (125 max dasharray)
    // 0 is full gauge (F1 = 1.0), 125 is empty gauge (F1 = 0)
    const strokeOffset = 125 - (simulatedF1 * 125);
    document.getElementById("gauge-fill-arc").style.strokeDashoffset = strokeOffset;
    
    // Update interpretations text
    const interpreter = document.getElementById("sens-interpretation");
    let txt = "";
    if (simulatedF1 > 0.55) {
        txt = `Mükemmel seviye (${simulatedF1.toFixed(2)} F1). Pencere boyutu (w=${w}) ve küçük sembol alfabeleri, gürültüyü başarılı bir şekilde filtreleyerek en optimum kararları üretiyor.`;
    } else if (simulatedF1 > 0.25) {
        txt = `Kararlı seviye (${simulatedF1.toFixed(2)} F1). Normal çalışma alanlarının sınırları sembolik olarak dengeli bir şekilde ayrışmış durumda.`;
    } else {
        txt = `Zayıf Seviye (${simulatedF1.toFixed(2)} F1). Geniş alfabe boyutu veya büyük pencereler, sahte geçiş yolları üreterek anomali hassasiyetini düşürüyor.`;
    }
    interpreter.textContent = txt;
}
