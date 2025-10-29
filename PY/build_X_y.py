COUNT = 1024
from Models.fuzz2 import fuzz2
from PY.scapy_tools import get_packet_instance_by_name, pkt2field_names_bytes
from pathlib import Path
from scapy.all import *


def build_real_x_y(STATIC_KEYS, PARENTS):
    X, y, ps = [], [], []


    ScapyNames =  {p: [p] for p in STATIC_KEYS} | {'ModBusTCP': ['ModbusADURequest', 'ModbusADUResponse']}
    ScapyFields = {}

    for P, Par in zip(STATIC_KEYS, PARENTS):
        print('>>>>>>>', P)
        root_dir = Path(f'Headers/{P}') #Path(f'../apre-benchmark-database/src/APREdatabase/Protocols/{P}')
        pcap_files = list(root_dir.rglob('*.pcap*'))

        for file in pcap_files:
            print(file)
            pkts = rdpcap(str(file), count=1024)  # convert Path to string
            #if len(pkts)<1024:
            print('NUM PKTS', len(pkts))
            #  continue


            first_names = False
            i = 0
            while not first_names:
                print('here',ScapyNames[P], Par, pkts[i])
                first_names, pkt_bytes = pkt2field_names_bytes(ScapyNames[P], Par, pkts[i])
                i += 1

            ScapyFields[P] = first_names
            trace = [pkt_bytes]
            print('first fields', first_names)

            for p in pkts[i:]:
                f_names, pkt_bytes = pkt2field_names_bytes(ScapyNames[P], Par, p)
                if f_names == first_names:
                    assert f_names == first_names, (first_names, f_names, pkts[0])
                    trace.append(pkt_bytes)

                else:
                    #pkt2field returned false
                    continue

            print('NUM PARSED', len(trace))
            assert len(trace) > 0.5*len(pkts)
            
            X.append(trace)
            y.append(0) #0 is true and 1 is fake    
            ps.append(P)
    
    return X, y, ps

from Models.ImageGenModel import CNNPacketImageGenerator
import torch
from scapy_tools import pkt2field_types


def build_fake_x_y(p_count_dict, STATIC_KEYS, PARENTS, generator='Scapy', seq_len = 256, FIELD_TYPE_VOCAB=[], max_len=10):
    assert generator in ['Scapy', 'Markov', 'CNN']
    X, y, ps = [], [], []
    ScapyNames =  {p: [p] for p in STATIC_KEYS} | {'ModBusTCP': ['ModbusADURequest', 'ModbusADUResponse']}
    field_type_to_idx = {ftype: i+1 for i, ftype in enumerate(FIELD_TYPE_VOCAB)} | {"None" : 0}

    for P, Par in zip(STATIC_KEYS, PARENTS):
        Par = PARENTS[STATIC_KEYS.index(P)]
        layer = ScapyNames[P][0]

        if generator=='CNN':
            # Assuming model class is defined exactly the same
            model = CNNPacketImageGenerator(field_vocab_size=len(FIELD_TYPE_VOCAB)+1)  # default args should be ok
            model.load_state_dict(torch.load(f"Models/{P}_weights.pth"))
            model.eval()


        for i in range(p_count_dict[P]):
            trace = []

            if generator in ['Scapy', 'Markov']:
                #### FUZZ GENERATORS
                pkt = get_packet_instance_by_name(layer)
                print(P, layer, pkt)
                p_prev = False
                for i in range(seq_len):
                    #need fresh instance
                    pkt = fuzz2(get_packet_instance_by_name(layer), p_prev=p_prev)
                    if generator == 'Markov':
                        p_prev = pkt
                    else:
                        p_prev = None #so fuzz2 acts as fuzz

                    f_names, pkt_bytes = pkt2field_names_bytes(ScapyNames[P], Par, pkt)
                    if f_names:
                        trace.append(pkt_bytes)

                    else:
                        #pkt2field returned false
                        continue

            elif generator == 'CNN':
                fts = pkt2field_types(ScapyNames[P])
                fts =  fts[:max_len] + ["None"] * max(0, max_len - len(fts))
                field_ids = torch.tensor([field_type_to_idx[t] for t in fts], dtype=torch.long).unsqueeze(0)
                #print(field_ids)
                pred = model(field_ids)[0]  # shape: [1, 1, 256, 32]
                trace = packet_image_to_bytes(pred)
                #print(f'{trace=}')
                

            print('NUM PARSED', len(trace))
            assert len(trace) > 0.9*seq_len
            
            X.append(trace)
            y.append(1)
            ps.append(P)

    return X, y, ps


import torch

def packet_image_to_bytes(pred_tensor):
    """
    Convert model output tensor to a list of packets (bytes objects).
    Each row is treated as one packet. Pixels with value -255 are ignored.
    """
    img = pred_tensor.squeeze().cpu()  # (256, 32)

    # Clamp to [0, 255], keep -255 as-is
    img = torch.where(img == -255, img, torch.clamp(img, 0, 255))

    packets = []
    for row in img:
        if (row == -255).all():
            continue  # skip padded rows

        valid_bytes = row[row != -255].round().to(torch.uint8).tolist()
        pkt = bytes(valid_bytes)
        packets.append(pkt)

    return packets