
# download project files from flywheel.io platform to .tar file

import time
import argparse
import flywheel

#!python3 fw_project_tar_get.py --api_key $FLYWHEEL_TOKEN --project_id $FLYWHEEL_PROJECT_ID

def parse_args():
    parser = argparse.ArgumentParser(description='Flywheel: download files from project into a .tar file.')
    parser.add_argument('--api_key', help='user api key')
    parser.add_argument('--project_id', help='project id')
    args = parser.parse_args()
    return args

def fw_project_tar_download(api_key, project_id):
    fw = flywheel.Client(api_key)
    project = fw.get_project(project_id)
    tic = time.time()
    fw.download_tar(project, '../data/project-files.tar', exclude_types=['nifti', 'dicom'])
    toc = (time.time()-tic)
    print('time elapsed: %.2f min' % (float(toc)/60)) # took ~18 minutes last run on 2022/11/30

def main(args):
    api_key = args.api_key
    project_id = args.project_id
    fw_project_tar_download(api_key, project_id)

if __name__ == "__main__":
    args = parse_args()
    main(args)
