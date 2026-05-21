python send_runpod_request.py --user-id alice --workflow test_input.json


python send_runpod_request.py --user-id alice --workflow workflows/flux_dev_t2i.json --set prompt="{{prompt}}"


python send_runpod_request.py --user-id alice --workflow workflows/flux_dev_t2i.json --set prompt="{{prompt}}" --set width={{width}} --set height={{height}}