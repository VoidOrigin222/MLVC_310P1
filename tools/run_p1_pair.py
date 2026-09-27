"""Run bounded two-device acceptance, keeping configs and raw logs per run."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def ssh(host, script):
    return ['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
            'bash -lc ' + shlex.quote(script)]


def run(host, script):
    return subprocess.check_output(ssh(host, script), stderr=subprocess.STDOUT,
                                   timeout=30).decode('utf-8', errors='replace')


def metrics(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def validate_result(result, frames):
    em, dm = result['encoder'], result['decoder']
    if result['encoder_exit'] != 0 or result['decoder_exit'] != 0:
        raise RuntimeError('process failed')
    if em.get('encode') != 'ok' or dm.get('decode') != 'ok':
        raise RuntimeError('missing success marker')
    if int(em['frames']) != frames or int(dm['frames']) != frames:
        raise RuntimeError('frame count mismatch')
    if em['bitstream_bytes'] != dm['bitstream_bytes']:
        raise RuntimeError('payload size mismatch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder-host', required=True, help='SSH host or alias for the encoder P1')
    ap.add_argument('--decoder-host', required=True, help='SSH host or alias for the decoder P1')
    ap.add_argument('--decoder-ip', required=True, help='decoder address reachable from the encoder')
    ap.add_argument('--remote-root', required=True,
                    help='remote data/model root containing data/ and mlvc* model directories')
    ap.add_argument('--deploy-dir', required=True, help='remote checkout directory containing build/')
    ap.add_argument('--cann-env', default='/usr/local/Ascend/cann/set_env.sh',
                    help='remote CANN set_env.sh path')
    ap.add_argument('--resolution', choices=['720p', '1080p'], default='720p')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--format', choices=['none', 'png'], default='none')
    ap.add_argument('--transport', choices=['udp', 'rtp'], default='udp')
    ap.add_argument('--port', type=int, default=39191)
    ap.add_argument('--ltr-period', type=int, default=0)
    ap.add_argument('--entropy-workers', type=int, default=2)
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    remote_root = args.remote_root.rstrip('/')
    deploy_dir = args.deploy_dir.rstrip('/')
    cann_setup = 'source ' + shlex.quote(args.cann_env)
    tag = '{}-{}-{}-{}-{}'.format(args.resolution, args.frames, args.transport,
                                   args.format, time.strftime('%H%M%S'))
    logs = Path(__file__).resolve().parents[1] / 'acceptance' / tag
    logs.mkdir(parents=True)
    remote = deploy_dir + '/acceptance/' + tag
    encoder_model = remote_root + ('/mlvc720p/1280x720/manifest_310P1.json' if args.resolution == '720p'
                                   else '/mlvc1080p/manifest_310P1.json')
    decoder_model = encoder_model if args.resolution == '720p' else remote_root + '/mlvc1080p/manifest_decoder_310P1.json'
    pipeline = '\n[pipeline]\nstream_workers = 1\nqueue_capacity = 3\nframe_buffer_slots = 3\nentropy_workers = {}\ngraph_packet_capacity = 2\n[model]\nmanifest = "{}"\n'
    pacing_rate = 1000000 if args.transport == 'rtp' else 10000000
    burst_bytes = 1200 if args.transport == 'rtp' else 4160
    enc = ('mode = "encode"\ninput_frame_dir = "' + remote_root + '/data/preprocessed_frames/614lab_' + args.resolution + '_fp16"\n'
           'device = 0\nframe_num = ' + str(args.frames) + '\nqp = 2\ngop = 96\nreset_interval = 32\n'
           'ltr_start_idx = 8\nltr_period = ' + str(args.ltr_period) + '\nltr_qp_shift = 8\nexecution_profile = "pipeline-v1"\n'
           'output_transport_mode = "' + args.transport + '"\n'
           'output_transport_host = "' + args.decoder_ip + '"\noutput_transport_port = ' + str(args.port) + '\n'
           'output_transport_pacing_rate_bps = ' + str(pacing_rate) + '\noutput_transport_max_burst_bytes = ' + str(burst_bytes) + '\n'
           + pipeline.format(1, encoder_model))
    profile_line = ('profile_output = "' + remote + '/decoder.trace.json"\n'
                    if args.profile else '')
    dec = ('mode = "decode"\ndevice = 0\nframe_num = -1\nformat = "' + args.format + '"\n' +
           profile_line +
           'output = "' + remote + '/reconstruction"\ninput_transport_port = ' + str(args.port) + '\n'
           'input_transport_mode = "' + args.transport + '"\n'
           'execution_profile = "pipeline-v1"\n' +
           pipeline.format(args.entropy_workers, decoder_model))
    for host, name, config in [(args.encoder_host, 'encoder', enc), (args.decoder_host, 'decoder', dec)]:
        local = logs / (name + '.toml')
        local.write_text(config)
        run(host, 'mkdir -p ' + shlex.quote(remote))
        subprocess.run(['scp', str(local), host + ':' + remote + '/' + name + '.toml'], check=True, timeout=30)
    # Check the port before starting, then wait for the actual bind (no guessed startup sleep).
    # Avoid nested awk quoting here: the local runner is Windows PowerShell
    # while the remote command is executed through bash -lc.
    check_port = "grep -i :%04X /proc/net/udp || true" % args.port
    if run(args.decoder_host, check_port).strip():
        raise RuntimeError('test UDP port already occupied')
    print('START ' + tag, flush=True)
    with (logs / 'decoder.log').open('w') as out:
        decoder = subprocess.Popen(ssh(args.decoder_host, cann_setup + '; cd ' + shlex.quote(deploy_dir) +
                    '; exec timeout 180s ./build/mlvc_decode --config ' + shlex.quote(remote + '/decoder.toml')),
                    stdout=out, stderr=subprocess.STDOUT)
        try:
            # Model loading on the decoder can take several seconds.  A fixed
            # grace period avoids repeated SSH probes competing with the
            # long-lived decoder SSH session on the lab host.
            time.sleep(20)
            if decoder.poll() is not None:
                raise RuntimeError('decoder exited before encoder start; inspect ' +
                                   str(logs / 'decoder.log'))
            print('Decoder bound; starting encoder', flush=True)
            with (logs / 'encoder.log').open('w') as encout:
                encoder = subprocess.run(ssh(args.encoder_host, cann_setup + '; cd ' + shlex.quote(deploy_dir) +
                             '; exec timeout 120s ./build/mlvc_encode --config ' + shlex.quote(remote + '/encoder.toml')),
                             stdout=encout, stderr=subprocess.STDOUT, timeout=135)
            print('Encoder exit: ' + str(encoder.returncode), flush=True)
        finally:
            # Wait for this bounded remote job even on failure; never kill unrelated jobs.
            try:
                decoder.wait(timeout=190)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait()
    em, dm = metrics(logs / 'encoder.log'), metrics(logs / 'decoder.log')
    result = {'encoder_exit': encoder.returncode, 'decoder_exit': decoder.returncode,
              'encoder': em, 'decoder': dm, 'remote_run': remote}
    (logs / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    validate_result(result, args.frames)
    if args.format == 'png':
        count = int(run(args.decoder_host, 'find ' + shlex.quote(remote + '/reconstruction') +
                        " -maxdepth 1 -type f -name 'im*.png' | wc -l").strip())
        if count != args.frames:
            raise RuntimeError('PNG count mismatch: ' + str(count))
    print('PASS ' + str(logs), flush=True)


if __name__ == '__main__':
    main()
