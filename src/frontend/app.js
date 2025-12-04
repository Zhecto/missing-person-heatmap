// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// State management
let appState = {
    dataLoaded: false,
    dataCleaned: false,
    clusteringDone: false,
    predictionReady: false
};

// Utility Functions
function showToast(message, type = 'info') {
    const toast = new bootstrap.Toast(document.getElementById('notificationToast'));
    const toastBody = document.getElementById('toastMessage');
    toastBody.textContent = message;
    toastBody.className = `toast-body bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} text-white`;
    toast.show();
}

function showLoading() {
    const overlay = document.createElement('div');
    overlay.className = 'spinner-overlay';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = '<div class="spinner-border text-light" role="status"><span class="visually-hidden">Loading...</span></div>';
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.remove();
}

function updateStatus(status) {
    appState = { ...appState, ...status };
    
    // Update status bar
    updateStatusBadge('status-loaded', status.data_loaded);
    updateStatusBadge('status-cleaned', status.data_cleaned);
    updateStatusBadge('status-clustering', status.clustering_done);
    updateStatusBadge('status-prediction', status.prediction_ready);
    
    // Update record counts
    document.getElementById('recordCount').textContent = status.total_records || 0;
    document.getElementById('cleanRecordCount').textContent = status.clean_records || 0;
    
    // Enable/disable buttons
    document.getElementById('cleanBtn').disabled = !status.data_loaded;
    document.getElementById('clusterBtn').disabled = !status.data_cleaned;
    document.getElementById('findOptimalBtn').disabled = !status.data_cleaned;
    document.getElementById('predictBtn').disabled = !status.data_cleaned;
    document.getElementById('trendsBtn').disabled = !status.data_cleaned;
    document.getElementById('generateHeatmapBtn').disabled = !status.data_cleaned;
    document.getElementById('generateChartsBtn').disabled = !status.data_cleaned;
}

function updateStatusBadge(elementId, active) {
    const element = document.getElementById(elementId);
    const badge = element.querySelector('.badge');
    const icon = element.querySelector('i');
    
    if (active) {
        badge.className = 'badge bg-success';
        badge.textContent = 'Complete';
        icon.style.color = '#28a745';
        element.classList.add('active');
    } else {
        badge.className = 'badge bg-secondary';
        badge.textContent = 'Pending';
        icon.style.color = '#6c757d';
        element.classList.remove('active');
    }
}

function displayResult(elementId, data, type = 'info') {
    const element = document.getElementById(elementId);
    const classes = type === 'success' ? 'result-box success' : type === 'error' ? 'result-box error' : 'result-box';
    
    let html = `<div class="${classes}">`;
    
    if (typeof data === 'string') {
        html += `<p>${data}</p>`;
    } else if (data.status) {
        html += `<h6>${data.status === 'success' ? '✓' : '✗'} ${data.message || ''}</h6>`;
        if (data.report) {
            html += `<pre style="max-height: 200px; overflow-y: auto; font-size: 0.85rem;">${data.report}</pre>`;
        }
        if (data.summary) {
            html += `<p><strong>Summary:</strong> ${JSON.stringify(data.summary, null, 2)}</p>`;
        }
    }
    
    html += '</div>';
    element.innerHTML = html;
}

function createTable(data, columns) {
    if (!data || data.length === 0) return '<p>No data available</p>';
    
    let html = '<div class="table-responsive"><table class="table table-striped table-hover">';
    html += '<thead class="table-dark"><tr>';
    
    // Headers
    columns.forEach(col => {
        html += `<th>${col.replace('_', ' ').toUpperCase()}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Rows
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            html += `<td>${value !== null && value !== undefined ? value : '-'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

// API Functions
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/status`);
        const status = await response.json();
        updateStatus(status);
    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showToast('Please select a file', 'error');
        return;
    }
    
    showLoading();
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Data uploaded successfully', 'success');
            displayResult('uploadResult', data, 'success');
            await checkStatus();
            await loadSummary();
        } else {
            showToast('Upload failed', 'error');
            displayResult('uploadResult', data, 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Upload failed: ' + error.message, 'error');
        displayResult('uploadResult', `Error: ${error.message}`, 'error');
    }
}

async function useDemoData() {
    showLoading();
    
    try {
        // Generate demo data using the backend
        const response = await fetch(`${API_BASE_URL}/`);
        const info = await response.json();
        
        hideLoading();
        showToast('Demo mode: Please run demo.py first to generate sample data', 'warning');
        displayResult('uploadResult', 'To use demo data, run: python demo.py', 'warning');
    } catch (error) {
        hideLoading();
        showToast('Failed to access API. Make sure the backend is running.', 'error');
    }
}

async function cleanData() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/clean`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Data cleaned successfully', 'success');
            displayResult('cleanResult', data, 'success');
            await checkStatus();
            await loadSummary();
        } else {
            showToast('Cleaning failed', 'error');
            displayResult('cleanResult', data, 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Cleaning failed: ' + error.message, 'error');
    }
}

async function loadSummary() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/summary`);
        const data = await response.json();
        
        // Display summary
        document.getElementById('summaryTotal').textContent = data.total_records || 0;
        document.getElementById('summaryLocations').textContent = data.locations?.unique_barangays || 0;
        
        if (data.date_range) {
            const dateRange = `${data.date_range.earliest} to ${data.date_range.latest}`;
            document.getElementById('summaryDateRange').textContent = dateRange;
        }
        
        if (data.gender_distribution) {
            const genderHtml = Object.entries(data.gender_distribution)
                .map(([gender, count]) => `<small>${gender}: ${count}</small><br>`)
                .join('');
            document.getElementById('summaryGender').innerHTML = genderHtml;
        }
        
        document.getElementById('dataSummary').style.display = 'block';
    } catch (error) {
        console.error('Failed to load summary:', error);
    }
}

async function runClustering() {
    const algorithm = document.getElementById('clusterAlgo').value;
    const requestData = {
        algorithm: algorithm
    };
    
    if (algorithm === 'kmeans') {
        requestData.n_clusters = parseInt(document.getElementById('numClusters').value);
    } else {
        requestData.eps = parseFloat(document.getElementById('epsValue').value);
        requestData.min_samples = parseInt(document.getElementById('minSamples').value);
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/analysis/clustering`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Clustering completed', 'success');
            
            // Display results
            let resultHtml = `<div class="result-box success">`;
            resultHtml += `<h6>✓ ${data.algorithm.toUpperCase()} Clustering Complete</h6>`;
            resultHtml += `<p><strong>Number of Clusters:</strong> ${data.n_clusters}</p>`;
            
            if (data.evaluation_metrics) {
                resultHtml += `<p><strong>Silhouette Score:</strong> ${data.evaluation_metrics.silhouette_score?.toFixed(3) || 'N/A'}</p>`;
            }
            
            resultHtml += `</div>`;
            displayResult('clusterResult', resultHtml);
            
            // Show results tables
            if (data.cluster_statistics) {
                document.getElementById('clusterStatsTable').innerHTML = createTable(
                    data.cluster_statistics,
                    ['Cluster', 'Size', 'Percentage', 'Top_Location']
                );
            }
            
            if (data.target_groups) {
                document.getElementById('targetGroupsTable').innerHTML = createTable(
                    data.target_groups,
                    ['Cluster', 'Description', 'Size', 'Dominant_Gender', 'Dominant_Age_Group']
                );
            }
            
            document.getElementById('resultsSection').style.display = 'block';
            await checkStatus();
        } else {
            showToast('Clustering failed', 'error');
            displayResult('clusterResult', data, 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Clustering failed: ' + error.message, 'error');
    }
}

async function findOptimalClusters() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/analysis/optimal-clusters?min_k=2&max_k=10`);
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Optimal K analysis complete', 'success');
            
            let resultHtml = `<div class="result-box success">`;
            resultHtml += `<h6>✓ Optimal K Analysis</h6>`;
            resultHtml += `<table class="table table-sm">`;
            resultHtml += `<thead><tr><th>K</th><th>Silhouette</th><th>Inertia</th></tr></thead><tbody>`;
            
            for (const [k, metrics] of Object.entries(data.results)) {
                resultHtml += `<tr><td>${k}</td><td>${metrics.silhouette_score.toFixed(3)}</td><td>${metrics.inertia.toFixed(0)}</td></tr>`;
            }
            
            resultHtml += `</tbody></table></div>`;
            document.getElementById('clusterResult').innerHTML = resultHtml;
        } else {
            showToast('Analysis failed', 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Analysis failed: ' + error.message, 'error');
    }
}

async function runPrediction() {
    const targetYear = parseInt(document.getElementById('targetYear').value);
    const topN = parseInt(document.getElementById('topN').value);
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/analysis/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_year: targetYear, top_n: topN })
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Prediction completed', 'success');
            
            let resultHtml = `<div class="result-box success">`;
            resultHtml += `<h6>✓ Prediction Model Trained</h6>`;
            resultHtml += `<p><strong>Target Year:</strong> ${data.target_year}</p>`;
            resultHtml += `<p><strong>R² Score:</strong> ${data.training_metrics.test_r2.toFixed(3)}</p>`;
            resultHtml += `<p><strong>RMSE:</strong> ${data.training_metrics.test_rmse.toFixed(2)}</p>`;
            resultHtml += `</div>`;
            displayResult('predictResult', resultHtml);
            
            // Display predictions
            if (data.predictions) {
                document.getElementById('predictionsTable').innerHTML = createTable(
                    data.predictions,
                    ['Barangay District', 'Predicted_Cases', 'Prev_Year_Count', 'Latitude', 'Longitude']
                );
            }
            
            document.getElementById('resultsSection').style.display = 'block';
            await checkStatus();
        } else {
            showToast('Prediction failed', 'error');
            displayResult('predictResult', data, 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Prediction failed: ' + error.message, 'error');
    }
}

async function analyzeTrends() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/analysis/trends`);
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Trend analysis complete', 'success');
            
            let resultHtml = `<div class="result-box success">`;
            resultHtml += `<h6>✓ Trend Analysis</h6>`;
            
            if (data.seasonal_patterns) {
                resultHtml += `<p><strong>Peak Month:</strong> ${data.seasonal_patterns.peak_month}</p>`;
                resultHtml += `<p><strong>Lowest Month:</strong> ${data.seasonal_patterns.lowest_month}</p>`;
            }
            
            resultHtml += `</div>`;
            document.getElementById('predictResult').innerHTML = resultHtml;
        } else {
            showToast('Trend analysis failed', 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Trend analysis failed: ' + error.message, 'error');
    }
}

async function generateHeatmap() {
    showLoading();
    
    try {
        const includeClusters = appState.clusteringDone;
        const response = await fetch(`${API_BASE_URL}/api/visualization/heatmap?include_clusters=${includeClusters}`);
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Heatmap generated', 'success');
            displayResult('visualizationResult', `Heatmap saved to: ${data.file_path}`, 'success');
            
            // Try to load heatmap in iframe (won't work with file:// but shows intent)
            document.getElementById('heatmapPreview').style.display = 'block';
            document.getElementById('heatmapFrame').src = data.file_path;
        } else {
            showToast('Heatmap generation failed', 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Heatmap generation failed: ' + error.message, 'error');
    }
}

async function generateCharts() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/visualization/charts`);
        const data = await response.json();
        hideLoading();
        
        if (data.status === 'success') {
            showToast('Charts generated', 'success');
            displayResult('visualizationResult', `${data.message}. Saved to: ${data.output_directory}`, 'success');
        } else {
            showToast('Chart generation failed', 'error');
        }
    } catch (error) {
        hideLoading();
        showToast('Chart generation failed: ' + error.message, 'error');
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Check initial status
    checkStatus();
    
    // Upload
    document.getElementById('uploadBtn').addEventListener('click', uploadFile);
    document.getElementById('useDemoBtn').addEventListener('click', useDemoData);
    
    // Preprocessing
    document.getElementById('cleanBtn').addEventListener('click', cleanData);
    
    // Clustering
    document.getElementById('clusterAlgo').addEventListener('change', function(e) {
        if (e.target.value === 'kmeans') {
            document.getElementById('kmeansOptions').style.display = 'block';
            document.getElementById('dbscanOptions').style.display = 'none';
        } else {
            document.getElementById('kmeansOptions').style.display = 'none';
            document.getElementById('dbscanOptions').style.display = 'block';
        }
    });
    
    document.getElementById('clusterBtn').addEventListener('click', runClustering);
    document.getElementById('findOptimalBtn').addEventListener('click', findOptimalClusters);
    
    // Prediction
    document.getElementById('predictBtn').addEventListener('click', runPrediction);
    document.getElementById('trendsBtn').addEventListener('click', analyzeTrends);
    
    // Visualization
    document.getElementById('generateHeatmapBtn').addEventListener('click', generateHeatmap);
    document.getElementById('generateChartsBtn').addEventListener('click', generateCharts);
});
