import os
from glob import glob
from src.net import Mode
from src.flownet2.flownet2 import FlowNet2

FLAGS = None


def main():
    folder = 'HMB_1'
    img_dir = [y for x in os.walk('/media/astra/1TB/Data/Ch2_002/{}/resized'.format(folder)) for y in glob(os.path.join(x[0], '*.png'))]
    img_dir.sort()

    # Create a new network
    net = FlowNet2(mode=Mode.TEST, debug=False)

    out_dir = '/media/astra/1TB/Data/Ch2_002/{}/of/'.format(folder)


    for i in range(100):
        net.test(checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                 input_a_path=img_dir[i],
                 input_b_path=img_dir[i+1],
                 out_path=out_dir + 'opflow_{0:06d}.png'.format(i))

        if i % 10 == 0:
            print("Total {} images, now {}th image".format(len(img_dir), i))


if __name__ == '__main__':
    main()