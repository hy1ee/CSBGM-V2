# When Training , NICE model loss suddenly crashes, Manually supplementing loss images

import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def main(hparams):

    if not os.path.exists(hparams.nice_save_dir):
        os.makedirs(hparams.nice_save_dir)

    if not os.path.exists(hparams.realnvp_save_dir):
        os.makedirs(hparams.realnvp_save_dir)

    # NICE
    plt.figure(figsize=(12, 8))  
    NICE_Loss = [-16746.068359375, -3867.242919921875, -1272.317138671875, -56.36056137084961, 636.266357421875, 1029.37109375,
                 1305.39208984375, 1481.4688720703125, 1599.0286865234375, 1677.3463134765625, 1735.6859130859375, 1779.184326171875,
                 1812.224365234375, 1838.0648193359375, 1859.3809814453125, 1892.3582763671875, 1906.8154296875, 1915.918701171875,
                 1924.4722900390625] # , -3.590161925775819e+18, -1793656994398208.0

    x = range(len(NICE_Loss))
    plt.plot(x, NICE_Loss)

    plt.title('NICE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig('./%s/loss_comparison.png' % (hparams.nice_save_dir))

    # RealNVP
    plt.figure(figsize=(12, 8))  
    RealNVP_Loss = [5129.25048828125, 6266.57666015625, 6161.08203125, 6863.69873046875, 6916.056640625,
                 7420.77978515625, 8575.3876953125, 8842.1640625, 9009.884765625, 9166.640625,
                 9286.08203125, 9391.330078125, 9522.4384765625, 9624.458984375, 9735.7763671875,
                 9817.748046875, 9709.18359375, 10057.8017578125, 10135.3427734375, 10268.4794921875,
                 10339.646484375, 10385.1220703125, 8996.6279296875, 10181.8330078125, 10327.9375,
                 10614.197265625, 10704.529296875, 9606.5634765625, 8111.16015625, 9676.232421875,
                 9526.5, 8701.4462890625, 7386.81591796875, 7666.00830078125, 7074.234375, 
                 7586.484375, 8011.345703125, 8913.0615234375, 7293.09228515625, 7498.3056640625,
                 7360.97265625, 6974.19482421875, 7505.12060546875, 7529.40478515625, 7760.61669921875, 
                 7760.68994140625, 7384.97412109375, 7731.05419921875, 7778.181640625, 7688.15771484375,
                 7794.583984375]
    # 7667.83837890625, 7125.78564453125, 7244.78271484375

    x = range(len(RealNVP_Loss))
    plt.plot(x, RealNVP_Loss)

    plt.title('NICE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig('./%s/loss_comparison.png' % (hparams.realnvp_save_dir))



if __name__ == '__main__':
    PARSER = ArgumentParser()

    PARSER.add_argument('--nice_save_dir', type=str, default='./plot_loss/NICEResult', help='Model saving directory')
    PARSER.add_argument('--realnvp_save_dir', type=str, default='./plot_loss/RealNVPPoint', help='Model saving directory')
    HPARAMS = PARSER.parse_args()
    main(HPARAMS)

