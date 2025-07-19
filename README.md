AI-Powered Network Intrusion Detection System (NIDS)
This project is a full-stack, containerized, AI-driven Network Intrusion Detection System that leverages state-of-the-art machine learning and web technologies to make advanced network security both powerful and accessible.

Features
Real-time and PCAP-based intrusion detection using XGBoost and BiLSTM models

Modern web dashboard (Flask) for monitoring, analysis, and visualization

Docker-ready and portable

Works on Kali Linux (and other Linux distros with Python 3.12 support)

Quick Start: Installation on Kali Linux
1. Install Python 3.12
If your Kali Linux does not have Python 3.12, install it:

bash
sudo apt update
sudo apt install curl gpg gnupg2 software-properties-common apt-transport-https lsb-release ca-certificates
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev python3.12-full
(If you encounter issues with the PPA or packages, see this guide or consider building from source.)

2. Clone the Repository
bash
git clone <your-repo-url>
cd nids_project
3. Create and Activate a Virtual Environment
bash
python3.12 -m venv venv_tf
source venv_tf/bin/activate
4. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
5. (Optional) Install system dependencies for packet capture
TShark (required by PyShark):

bash
sudo apt install tshark
6. Run the Web Dashboard
bash
python dashboard/app.py
Access the dashboard at http://127.0.0.1:5000 in your browser.

Usage
Start/stop live monitoring or analyze PCAP files from the dashboard.

Select detection model (XGBoost or BiLSTM).

View real-time alerts, detection statistics, and visualizations.

Troubleshooting
ModuleNotFoundError:
Make sure your virtual environment is activated and all dependencies are installed.

TensorFlow install fails:
Double-check that you are using Python 3.12 (not 3.13+).

No intrusions detected:
Try lowering the detection threshold in the code or use a PCAP file with known attacks.

Permission errors with TShark:
You may need to run sudo usermod -aG wireshark $USER and restart your session.

Uninstall / Cleanup
To deactivate and remove your virtual environment:

bash
deactivate
rm -rf venv_tf