"""
Stylish Dashboard CSS and Components for Options Trading System
"""

# Enhanced CSS for stylish dashboard
DASHBOARD_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.main {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Header Styles */
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.main-header h1 {
    color: white;
    text-align: center;
    margin: 0;
    font-weight: 700;
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* Tab Styles */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.stTabs [data-baseweb="tab"] {
    height: 60px;
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    border-radius: 12px;
    color: white !important;
    font-weight: 600;
    font-size: 16px;
    border: none;
    padding: 0 25px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
}

/* Card Styles */
.metric-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    margin: 10px 0;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
}

.trading-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin: 15px 0;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

/* Button Styles */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}

.success-button {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
}

.danger-button {
    background: linear-gradient(135deg, #f44336 0%, #da190b 100%) !important;
}

.warning-button {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%) !important;
}

/* Alert Styles */
.stAlert {
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
}

/* Sidebar Styles */
.css-1d391kg {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
}

/* Metric Styles */
.css-1xarl3l {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Table Styles */
.stDataFrame {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

/* Input Styles */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
}

.stSelectbox > div > div > div {
    border-radius: 12px !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
}

/* Spinner Styles */
.stSpinner {
    border-top-color: #667eea !important;
}

/* Custom Animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}

/* Popup Modal Styles */
.popup-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(5px);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}

.popup-content {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
    padding: 40px;
    border-radius: 25px;
    max-width: 800px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 25px 80px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(20px);
    position: relative;
}

.popup-close {
    position: absolute;
    top: 15px;
    right: 20px;
    background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
    border: none;
    color: white;
    font-size: 20px;
    font-weight: bold;
    padding: 10px 15px;
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.3s ease;
}

.popup-close:hover {
    transform: scale(1.1);
    box-shadow: 0 5px 15px rgba(244, 67, 54, 0.4);
}

/* Live Data Indicators */
.live-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background: #4CAF50;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
    100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
}

.offline-indicator {
    background: #f44336;
    animation: none;
}

/* Trading Signal Cards */
.signal-card {
    background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%);
    border-left: 5px solid #4CAF50;
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(76, 175, 80, 0.2);
}

.signal-card.sell {
    background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
    border-left-color: #f44336;
    box-shadow: 0 4px 20px rgba(244, 67, 54, 0.2);
}

.signal-card.hold {
    background: linear-gradient(135deg, #fff3e0 0%, #fef7e0 100%);
    border-left-color: #ff9800;
    box-shadow: 0 4px 20px rgba(255, 152, 0, 0.2);
}

/* Progress Bars */
.progress-bar {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    overflow: hidden;
    height: 8px;
    margin: 10px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    transition: width 0.3s ease;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.status-success {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    color: white;
}

.status-warning {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    color: white;
}

.status-error {
    background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
    color: white;
}

.status-info {
    background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
    color: white;
}

/* Grid Layout */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.grid-item {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.grid-item:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
}

/* Dark Mode Compatibility */
@media (prefers-color-scheme: dark) {
    .metric-card {
        background: rgba(40, 44, 52, 0.95);
        color: white;
    }
    
    .popup-content {
        background: linear-gradient(135deg, rgba(40, 44, 52, 0.95) 0%, rgba(40, 44, 52, 0.85) 100%);
        color: white;
    }
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 14px;
        padding: 0 15px;
    }
    
    .dashboard-grid {
        grid-template-columns: 1fr;
        gap: 15px;
    }
    
    .popup-content {
        margin: 20px;
        padding: 25px;
        max-width: calc(100vw - 40px);
    }
}
</style>
"""

# JavaScript for popup functionality
POPUP_JS = """
<script>
// Popup Window Functions
function showPopup(content, title) {
    const overlay = document.createElement('div');
    overlay.className = 'popup-overlay';
    overlay.onclick = function(e) {
        if (e.target === overlay) {
            closePopup(overlay);
        }
    };
    
    const popup = document.createElement('div');
    popup.className = 'popup-content';
    popup.innerHTML = `
        <button class="popup-close" onclick="closePopup(this.closest('.popup-overlay'))">&times;</button>
        <h2 style="margin-top: 0; color: #1e3c72; font-weight: 600;">${title}</h2>
        <div>${content}</div>
    `;
    
    overlay.appendChild(popup);
    document.body.appendChild(overlay);
    
    // Animation
    popup.style.transform = 'scale(0.8)';
    popup.style.opacity = '0';
    setTimeout(() => {
        popup.style.transform = 'scale(1)';
        popup.style.opacity = '1';
        popup.style.transition = 'all 0.3s ease';
    }, 10);
}

function closePopup(overlay) {
    const popup = overlay.querySelector('.popup-content');
    popup.style.transform = 'scale(0.8)';
    popup.style.opacity = '0';
    setTimeout(() => {
        overlay.remove();
    }, 300);
}

// Live Data Status Update
function updateLiveStatus(isLive) {
    const indicators = document.querySelectorAll('.live-indicator');
    indicators.forEach(indicator => {
        if (isLive) {
            indicator.classList.remove('offline-indicator');
        } else {
            indicator.classList.add('offline-indicator');
        }
    });
}

// Auto-refresh functionality
function startAutoRefresh() {
    setInterval(() => {
        // Trigger Streamlit rerun for live data
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: Date.now()}, '*');
    }, 30000); // 30 seconds
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    startAutoRefresh();
});

// Smooth scroll for internal links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({behavior: 'smooth'});
        }
    }
});
</script>
"""

def get_status_badge(status: str, text: str) -> str:
    """Generate status badge HTML"""
    return f'<span class="status-badge status-{status}">{text}</span>'

def get_live_indicator(is_live: bool = True) -> str:
    """Generate live indicator HTML"""
    class_name = "live-indicator" if is_live else "live-indicator offline-indicator"
    return f'<span class="{class_name}"></span>'

def create_signal_card(signal_type: str, action: str, confidence: float, details: str) -> str:
    """Create signal card HTML"""
    action_class = action.lower()
    return f"""
    <div class="signal-card {action_class}">
        <h4 style="margin: 0 0 10px 0; color: #1e3c72;">{signal_type}</h4>
        <p style="margin: 0; font-size: 18px; font-weight: 600;">{action}</p>
        <p style="margin: 5px 0; color: #666;">Confidence: {confidence:.2%}</p>
        <p style="margin: 10px 0 0 0; font-size: 14px; color: #555;">{details}</p>
    </div>
    """

def create_progress_bar(percentage: float, label: str = "") -> str:
    """Create progress bar HTML"""
    return f"""
    <div style="margin: 10px 0;">
        {f'<label style="font-size: 14px; color: #666; margin-bottom: 5px; display: block;">{label}</label>' if label else ''}
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percentage}%;"></div>
        </div>
        <span style="font-size: 12px; color: #888;">{percentage:.1f}%</span>
    </div>
    """