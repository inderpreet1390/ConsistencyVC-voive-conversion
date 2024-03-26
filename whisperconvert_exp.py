import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import soundfile as sf
import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="/content/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="/content/G_cvc-whispers-three-emo-loss.pth", help="path to pth file")
    parser.add_argument("--outdir", type=str, default="/content/output", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)


    src_wavs=[r"/content/730_359_000004_000001_.wav"]
    tgt_wavs=[r"/content/sample-000000.wav"]
    
    print("Processing text...")
    titles, srcs, tgts = [], [], []
    for src_wav in src_wavs:
        for tgt_wav in tgt_wavs:
            src_wav_name=os.path.basename(src_wav)[:-4]
            tgt_wav_name=os.path.basename(tgt_wav)[:-4]
            title="{}_to_{}".format(src_wav_name,tgt_wav_name)
            titles.append(title)
            srcs.append(src_wav)
            tgts.append(tgt_wav)
    print(srcs)
    print(tgts)
    print(titles)
    #import sys
    #sys.exit()
    """
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            print(rawline)
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)
    """

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            srcname,tgtname=title.split("to")
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            sf.write(os.path.join(args.outdir, f"{tgtname}.wav"), wav_tgt, hps.data.sampling_rate)
            #wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
     
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            sf.write(os.path.join(args.outdir, f"{srcname}.wav"), wav_src, hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            #c = utils.get_content(cmodel, wav_src)
            c_filename = src.replace(".wav", "whisper.pt.npy")
            #print(src, tgt,c_filename)
            #c = torch.load(c_filename)#.squeeze(0)
            import numpy as np
            c=torch.from_numpy(np.load(c_filename))
            c=c.transpose(1,0)
            c=c.unsqueeze(0)

            print(c.size(),mel_tgt.size())
            audio = net_g.infer(c.cuda(), mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)
            
