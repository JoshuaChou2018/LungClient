# Python client for lung segmentation online inference
# author: Joshua

import argparse
import re
import subprocess
import nibabel as nib
import numpy as np
import requests
import os
import gzip
import uuid
import time
import pyAesCrypt
import utils.dcm_np_converter as rescale
from utils import Functions
from utils import visualize_stl as stl
import requests

def check_available_servers(args):
    print(f'[+] retrieving server list')
    response = requests.get('https://raw.githubusercontent.com/JoshuaChou2018/LungClient/main/server_list.update')
    server_list = response.text.split('\n')
    response_list = []
    for host in server_list:
        try:
            output = subprocess.check_output(['ping', '-c', '1', '-q', host])
            output = output.decode('utf8')
            statistic = re.search(r'(\d+\.\d+/){3}\d+\.\d+', output).group(0)
            avg_time = re.findall(r'\d+\.\d+', statistic)[1]
            response_time = float(avg_time)
        except subprocess.CalledProcessError:
            response_time = 99999999
        response_list.append((host,response_time))
    response_list = sorted(response_list, key = lambda x: x[1],reverse=True)
    for r in response_list:
        print(f'{r[0]}\tping: {r[1]} ms')

def run_online_inference(args):

    dict_dcm = args.dict_dcm
    dict_save_final = args.output_dir
    file_name = args.file_name
    private_key_path = args.pri_key
    public_key_path = args.pub_key
    temp_save_dir = args.temp_dir
    temp_file_name = uuid.uuid4().hex
    upload_file_path = f'{temp_save_dir}/{temp_file_name}.npy'
    encrypted_upload_file_path = upload_file_path + '.enc'
    processed_file_path = f'{temp_save_dir}/{temp_file_name}.processed.npy.gz'
    encrypted_processed_file_path = processed_file_path + '.enc'
    upload_file_name = upload_file_path.split('/')[-1]
    upload_file_namebody = upload_file_name.rstrip('.')[0]
    server = args.server

    def read_pem(file_path):
        with open(file_path, 'r') as f:
            key = f.readlines()[1:-1]
            key = [x.rstrip('\n') for x in key]
        return ''.join(key)

    start_time = time.time()

    if os.path.isdir(temp_save_dir) == False:
        os.makedirs(temp_save_dir)

    # process data
    print(f'[+] processing original ct data to rescaled_ct with shape (512,512,512)')
    rescaled_ct = rescale.dcm_to_spatial_signal_rescaled(dict_dcm, (-600, 1600))
    print(f'[+] rescaled_ct has shape {rescaled_ct.shape}, saving it now')
    with open(upload_file_path, 'wb') as f:
        np.save(f, rescaled_ct)
    print(f'[+] saved rescaled_ct to {upload_file_path}')

    # encryption
    print(f'[+] reading private key @{private_key_path}')
    print(f'[+] encrypting data with private key @{private_key_path} to file {upload_file_path}.enc')
    private_key = read_pem(private_key_path)
    pyAesCrypt.encryptFile(upload_file_path, encrypted_upload_file_path, private_key)
    print(f'[+] encryption finished')

    complete_server = f'http://{server}:5000/lung'
    print(f'[+] uploaded rescaled_ct @{encrypted_upload_file_path} to server {complete_server}, waiting for online inference (this step takes long, but should be no longer than 10 mins)')
    # test_response, status = requests.post('http://ws1.joshuachou.ink:5000/lung', files =[("ct", open("temp/test.npy.gz", "rb"))])
    response = requests.post(complete_server,
                             files=[("ct", open(encrypted_upload_file_path, "rb")),
                                    ('public_key', open(public_key_path, "rb"))])

    status_code = response.status_code
    print(f'[+] received response from the server, checking the response')

    if status_code == 200:
        try:
            response_json = response.json()
            print(f'[+] job rejected due to {response_json}')
        except:
            print('[+] received encrypted processed result from server')
            with open(f'{encrypted_processed_file_path}', 'wb') as w:
                w.write(response.content)
            print(f'[+] saved results to {encrypted_processed_file_path}')

            # decrypt the processed result
            print(f'[+] decrypt the processed result to {processed_file_path}')
            pyAesCrypt.decryptFile(encrypted_processed_file_path, processed_file_path, private_key)

            # load processed data
            print(f'[+] post-processing result @{processed_file_path} from server')
            load_path = os.path.join(f'{processed_file_path}')
            f = gzip.GzipFile(load_path, "r")
            tissue_seg_list = np.load(f)
            print(f'[+] processed segmentation result with shape {tissue_seg_list.shape}')

            print('[+] rescaling results to original shape')

            lung_original = rescale.undo_spatial_rescale(dict_dcm, tissue_seg_list[0])
            heart_original = rescale.undo_spatial_rescale(dict_dcm, tissue_seg_list[1])
            blood_original = rescale.undo_spatial_rescale(dict_dcm, tissue_seg_list[2])
            airway_original = rescale.undo_spatial_rescale(dict_dcm, tissue_seg_list[3])
            nodule_original = rescale.undo_spatial_rescale(dict_dcm, tissue_seg_list[4])

            if len(file_name) > 4 and file_name[-4] == '.':
                file_name = file_name[:-4]

            resolution_original = rescale.get_original_resolution(dict_dcm)

            print('[+] saving results to mha format')

            Functions.save_np_as_mha(lung_original, dict_save_final, 'lung_' + file_name, spacing=resolution_original)
            Functions.save_np_as_mha(heart_original, dict_save_final, 'heart_' + file_name, spacing=resolution_original)
            Functions.save_np_as_mha(blood_original, dict_save_final, 'blood-vessel_' + file_name,
                                         spacing=resolution_original)
            Functions.save_np_as_mha(airway_original, dict_save_final, 'airway_' + file_name, spacing=resolution_original)
            Functions.save_np_as_mha(nodule_original, dict_save_final, 'nodule_' + file_name, spacing=resolution_original)

            print('[+] saving results to stl format')

            stl.convert_mha_to_stl(os.path.join(dict_save_final, 'lung_' + file_name + '.mha'),
                                       os.path.join(dict_save_final, 'lung_' + file_name + '.stl'))
            stl.convert_mha_to_stl(os.path.join(dict_save_final, 'heart_' + file_name + '.mha'),
                                       os.path.join(dict_save_final, 'heart_' + file_name + '.stl'))
            stl.convert_mha_to_stl(os.path.join(dict_save_final, 'blood-vessel_' + file_name + '.mha'),
                                       os.path.join(dict_save_final, 'blood-vessel_' + file_name + '.stl'))
            stl.convert_mha_to_stl(os.path.join(dict_save_final, 'airway_' + file_name + '.mha'),
                                       os.path.join(dict_save_final, 'airway_' + file_name + '.stl'))
            stl.convert_mha_to_stl(os.path.join(dict_save_final, 'nodule_' + file_name + '.mha'),
                                       os.path.join(dict_save_final, 'nodule_' + file_name + '.stl'))

            print(f'[+] saved final results to {dict_save_final}')

            print(f'[+] job finished')

    elif status_code == 500:

        print(response.text)

    try:
        os.remove(upload_file_path)
        os.remove(encrypted_upload_file_path)
        os.remove(processed_file_path)
    except:
        pass

    print(f'[+] cleaning temp files @{upload_file_path} @{encrypted_upload_file_path} @{processed_file_path}')
    print(f"[+] total time: {time.time() - start_time} s")

def main():
    parser = argparse.ArgumentParser(prog='Lung')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('server')
    subparser.set_defaults(callback=check_available_servers)

    subparser = subparsers.add_parser('run')
    subparser.set_defaults(callback=run_online_inference)
    subparser.add_argument("--server", help="server", required=True)
    subparser.add_argument("--temp_dir", help = "path/to/temp/dir", default = 'temp')
    subparser.add_argument("--dict_dcm", help = "path/to/dict/dcm/data", required = True)
    subparser.add_argument("--output_dir", help = "path/to/save/output", required = True)
    subparser.add_argument("--file_name", help = "file name", required = True)
    subparser.add_argument("--pub_key", help = "path/to/public.pem", required = True)
    subparser.add_argument("--pri_key", help = "path/to/private.pem", required = True)

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()

    #Example
    # python run.py server
    # python run.py run --server ws1.joshuachou.ink --temp_dir temp --dict_dcm 'Example/data_1_2020-05-01/raw_data' --output_dir 'Example/data_1_2020-05-01/prediction' --file_name 'A01' --pub_key 'Example/public.pem' --pri_key 'Example/private.pem'
    # python run.py run --server ws1.joshuachou.ink --temp_dir temp --dict_dcm 'Example/data_1_2020-05-01/raw_data' --output_dir 'Example/data_1_2020-05-01/prediction' --file_name 'A01' --pub_key 'Example/fake_public.pem' --pri_key 'Example/private.pem'
