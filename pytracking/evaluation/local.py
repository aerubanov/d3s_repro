from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.mobiface_path = ''
    settings.network_path = '/home/anatoly/HDD/DS_Projects/d3s_repro/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = '/home/anatoly/HDD/DS_Projects/d3s_repro/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot16_path = 'DATA/vot2016/sequences'
    settings.vot18_path = ''
    settings.vot_path = ''

    return settings

