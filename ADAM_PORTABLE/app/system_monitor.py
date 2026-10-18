"""Low-frequency system metrics collector with optional psutil support."""

import logging
import threading
from typing import Callable, Dict

logger = logging.getLogger("ADAM.SystemMonitor")


def read_system_metrics() -> Dict[str, str]:
    """Read real system metrics; unavailable sensors are reported explicitly."""
    try:
        import psutil
    except ImportError:
        return {
            "cpu": "Unavailable",
            "memory": "Unavailable",
            "battery": "Unavailable",
            "temperature": "Unavailable",
        }

    cpu = f"{psutil.cpu_percent(interval=0.25):.0f}%"
    memory = f"{psutil.virtual_memory().percent:.0f}%"

    try:
        battery_info = psutil.sensors_battery()
    except (AttributeError, OSError, psutil.Error):
        battery_info = None
    if battery_info is None:
        battery = "Unavailable"
    elif battery_info.percent is None:
        battery = "Unavailable"
    else:
        charge = f"{battery_info.percent:.0f}%"
        power = "charging" if battery_info.power_plugged else "on battery"
        battery = f"{charge}, {power}"

    temperature = "Unavailable"
    try:
        sensors = psutil.sensors_temperatures() or {}
        values = [
            sensor.current
            for readings in sensors.values()
            for sensor in readings
            if sensor.current is not None
        ]
        if values:
            temperature = f"{max(values):.0f}°C"
    except (AttributeError, OSError, psutil.Error):
        logger.debug("CPU temperature sensors are not available.")

    return {
        "cpu": cpu,
        "memory": memory,
        "battery": battery,
        "temperature": temperature,
    }


class SystemMonitor:
    """Runs sensor collection off the UI thread at a modest interval."""

    def __init__(self, callback: Callable[[Dict[str, str]], None], interval: float = 5.0):
        self.callback = callback
        self.interval = max(2.0, interval)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="adam-system-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self.callback(read_system_metrics())
            except Exception:
                logger.exception("System metrics collection failed.")
            self._stop_event.wait(self.interval)
