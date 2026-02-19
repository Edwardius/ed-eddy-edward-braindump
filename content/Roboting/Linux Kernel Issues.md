# Unable to mount to root fs
This occurred because for some reason my current linux kernel got corrupted.

## Steps to check
1. Boot to linux via a usb stick
2. Double check that your disks exist (`lsblk`) and are readable (that means theres a booting issue and NOT a disk issue)
	1. If this is a disk issue and your disk is unreadable, LOL i can't help you. Blow on the disk or something
3. If your partition still exists, nice! You can actually mount to it and run commands

	```bash
	sudo mount /dev/nvme1n1p2 /mnt
	sudo mount /dev/nvme1n1p1 /mnt/boot/efi
	sudo mount --bind /dev /mnt/dev
	sudo mount --bind /proc /mnt/proc
	sudo mount --bind /sys /mnt/sys
	sudo chroot /mnt
	```

	Inside your usb in your pencil case, the ubuntu you have in there has a script for you to do this: `./eddyzhou/mount_to_partition.sh <partition>`

4.  Some helpful commands to diagnose the issue with the partition

```bash
# Verify fstab uses UUIDs
cat /etc/fstab

# Update GRUB and initramfs
update-grub
grub-install /dev/nvme1n1
update-initramfs -u -k all

# Is the kernel actually there? and what is it
ls /boot/vmlinuz*

# Is the initramfs there?
ls /boot/initrd*

# Check if the root UUID matches
blkid /dev/nvme1n1p2

# 4. Check what's in the initramfs (grab from ls /boot/vmlinuz*)
lsinitramfs /boot/initrd.img-6.17.0-14-generic | grep nvme
```

I had two kernels installed for some reason, I think it was an artifact of the past.

That being said, when i ran `lsinitramfs /boot/initrd.img-6.17.0-14-generic | grep nvme` I got nothing and  so that meant that my kernel was fucked. (corrupted)

To fix:
```bash
update-initramfs -c -k 6.17.0-14-generic
update-grub
exit
sudo reboot
```
