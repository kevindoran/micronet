# Doing everything with the standard API:
# https://cloud.google.com/tpu/docs/managing-vm-tpu-resources

# The CTPU API way:
# https://cloud.google.com/tpu/docs/ctpu-reference
# --disk-size-gb needs to allow for 150 GB of ImageNet data. 500 GB disk size is reccomended.
ctpu up --name=kdoran1 --tpu-size=v2-8 --disk-size-gb=500 --preemptible --project=micronet-kdoran --zone=us-central1-f
