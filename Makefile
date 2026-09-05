# Sync local code to remote VM with GPU (using 'vast' from ssh config)
VM=vast
DEST=~/minitorch
#
sync:
	rsync -avzP --exclude='.git/' --exclude='.opencode/' --exclude='build/' --exclude='data/' ./ $(VM):$(DEST)/

