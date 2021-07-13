from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = 'DATA/got10k/'
    settings.lasot_path = ''
    settings.mobiface_path = ''
    settings.network_path = 'wdir/checkpoints/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = 'pytracking/tracking_results/tracking_net'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = 'DATA/trackingnet/'
    settings.uav_path = ''
    settings.vot16_path = 'DATA/vot2016/sequences'
    settings.vot18_path = 'DATA/vot2018/sequences'
    settings.vot_path = ''

    return settings

