import socket

def check_internet(host="8.8.8.8", port=53, timeout=3):
    """
    Checks for internet connectivity by attempting to create a TCP socket 
    connection to a highly available host (Google DNS by default).
    
    Args:
        host (str): The IP address to connect to (default: Google DNS 8.8.8.8).
        port (int): The port to connect to (default: 53/TCP).
        timeout (int): The timeout in seconds (default: 3).
        
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        # Set the default timeout for the socket operation
        socket.setdefaulttimeout(timeout)
        # Create a socket object (AF_INET = IPv4, SOCK_STREAM = TCP)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Attempt to connect
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False

# Usage
if check_internet():
    print("Online: Enabling optional features.")
else:
    print("Offline: Skipping optional features.")