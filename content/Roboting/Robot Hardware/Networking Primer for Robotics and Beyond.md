## What is Networking?

At its simplest, **networking** is how two or more devices exchange data. Every network is built on three questions:
1. **Who am I?** (identity → IP/MAC address)
2. **Who are you?** (discovery → ARP, DNS, broadcast)
3. **How do we talk?** (protocols → Ethernet, TCP/UDP, HTTP, ROS2 DDS, etc.)

Networking can happen:
- Directly, over a cable (Ethernet, USB-C networking).
- Through infrastructure (routers, switches, Wi-Fi).
- Or virtually (VPNs, bridges, Docker networks).
## Core Concepts
### Layers (OSI Model)
- **Layer 1 – Physical:** Cables, fiber, Wi-Fi radio.
- **Layer 2 – Data Link:** MAC addresses, switches.
- **Layer 3 – Network:** IP addresses, routers, subnets.
- **Layer 4 – Transport:** TCP/UDP ports, reliability vs speed.
- **Layer 7 – Application:** ROS2 DDS, HTTP, SSH, NTP, etc.

When debugging, always ask: _which layer is broken?_
### Addresses
- **MAC address:** Unique hardware identifier (e.g., `00:B0:9D:1A:F0:0E`). Works at Layer 2.
- **IP address:** Logical identity (e.g., `10.8.0.18`). Works at Layer 3.
- **Port:** Application endpoint (e.g., TCP 22 = SSH). Works at Layer 4.
### Subnets
- Networks are divided into **subnets** (ranges of IPs).
- Defined by an address and a mask:
    - `10.8.0.1/16` → covers `10.8.0.0` to `10.8.255.255`.
    - `192.168.1.1/24` → covers `192.168.1.0` to `192.168.1.255`.
Devices can only talk directly if they are on the **same subnet**.
### DHCP vs Static IP
- **Static IP:** Manually assigned. Predictable but brittle if reused.
- **DHCP:** Dynamic Host Configuration Protocol. Device asks for an IP, server assigns one. Often paired with **static leases** (MAC → fixed IP).
- **Link-Local (169.254.x.x):** Fallback if DHCP fails and no static IP is set.
### Routing and Gateways
- If traffic is destined for another subnet, it is sent to a **gateway** (a router).
- Example: Robot sensors (10.8.0.x) talk locally to the robot PC, which routes internet traffic out via a second NIC.
### DNS
- Human-friendly names (like `google.com`) → IP addresses.
- Optional for sensors (often direct IP is fine).
## Typical Networking Patterns in Robotics
1. **Isolated Sensor Network** (I prefer this, as it's the only one I know how to do :P)
    - Robot PC acts as DHCP server.
    - All sensors are on a private subnet (`10.x.x.x`).
    - No internet exposure.
2. **Dual-Homed PC**
    - One NIC faces the internet (`enp6s0`).
    - Another NIC faces the sensor network (`enp7s0f1`).
    - Allows development SSH + stable sensor comms.
3. **Bridged/Switched Network**
    - Multiple PCs/sensors on one subnet via a switch.
    - DHCP server assigns addresses.
4. **NAT/Forwarding**
    - Robot PC routes sensor network traffic to the internet (optional).
## Common Tools
### Linux Commands
- `ip a` → show addresses.
- `ip link` → show NICs.
- `ip neigh` → ARP table (who’s connected).
- `ping <IP>` → test reachability.
- `traceroute <IP>` → path to destination.
- `tcpdump -i <iface>` → sniff packets.
- `dig <hostname>` → test DNS resolution.
### Config Tools
- **Netplan** → declarative config on Ubuntu servers.
- **NetworkManager (nmcli)** → dynamic config on desktops.
- **dnsmasq** → lightweight DHCP + DNS server.
- **iptables/nftables** → firewall and NAT rules.
## Debugging
1. **Physical layer:** Is the cable plugged in? Is the link light on?
2. **IP layer:** Do devices have IPs in the same subnet?
3. **DHCP:** Did the device request an IP? (`journalctl -u dnsmasq -f`)
4. **ARP:** Does `ip neigh` show the device’s MAC?
5. **Ping:** Can you ping the sensor?
6. **Application:** Can you open the driver (ROS2 node, SpinView, etc.)?
## Common Fuckups
- **169.254.x.x addresses:** Device couldn’t reach DHCP → stuck in fallback.
- **Multiple managers:** Netplan vs NetworkManager fighting for control.
- **Wrong MTU:** GigE cameras often need `mtu=9000`.
- **Overlapping subnets:** Internet NIC and sensor NIC must not be on the same subnet.
- **Firewall rules:** Block discovery protocols (e.g., GigE Vision discovery via broadcast).
## Practical Scenarios
- **GigE Camera bringup:**
    - Disable Persistent IP.
    - Enable DHCP.
    - Ensure PC runs DHCP server (`dnsmasq`).
- **GPS/IMU bringup:**
    - Default static IP may need to be changed.
    - Enable DHCP if supported.
- **ROS 2 multi-machine:**
    - All machines must share the same subnet and domain ID.
    - Ensure multicast works (DDS relies on it).