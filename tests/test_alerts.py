# tests/test_alerts.py

from utils.alerts import send_alert

def test_send_alert_logs(caplog):
    with caplog.at_level("INFO"):
        send_alert("Test Alert Message")
        assert any("Test Alert Message" in msg for msg in caplog.text.splitlines())

