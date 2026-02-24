# Sync local code to remote VM with GPU (using 'vast' from ssh config)
VM=vast
DEST=~/minitorch
#
sync:
	rsync -avzP --exclude='.git/' --exclude='build/' ./ $(VM):$(DEST)/
#
# sync:
# 	rsync -avzP --rsync-path=/usr/bin/rsync --exclude='.git/' --exclude='build/' ./ $(VM):$(DEST)/
#
#
# sync:
# 	tar czf - --exclude='.git' --exclude='build' . | ssh $(VM) "mkdir -p $(DEST) && tar xzf - -C $(DEST)"
