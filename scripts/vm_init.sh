gcloud config set compute/zone us-central1-f
gcloud config set project micronet-kdoran

virtualenv venv --python=python3
source ./venv/bin/activate
pip install -r ./micronet_lsyncd/requirements.txt
cd micronet_lsyncd
pip install -e .

