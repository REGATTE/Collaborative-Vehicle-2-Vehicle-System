import psutil
import time

def log_system_resources():
    while True:
        # CPU Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_usage}%")
        
        # Memory Usage
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}% (Used: {memory.used / 1e6:.2f} MB)")

        # Disk Usage
        disk = psutil.disk_usage('/')
        print(f"Disk Usage: {disk.percent}% (Used: {disk.used / 1e9:.2f} GB)")

        # Network Usage
        net_io = psutil.net_io_counters()
        print(f"Network Sent: {net_io.bytes_sent / 1e6:.2f} MB, Received: {net_io.bytes_recv / 1e6:.2f} MB")

        print("-" * 30)
        time.sleep(5)  # Log every 5 seconds

if __name__ == "__main__":
    log_system_resources()
