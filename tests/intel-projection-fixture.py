"""Extract bounded local Q6_K projections for the optional NPU numerical test."""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.gguf_reader import GGUFFile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    args=parser.parse_args()
    header=GGUFFile(args.model)
    if header.metadata.get('general.architecture')!='qwen35moe':
        parser.error('Expected the pinned Qwen35MoE geometry')
    tensors=[next(t for t in header.tensors if t.name==name)
             for name in ('blk.0.attn_qkv.weight','blk.0.attn_gate.weight')]
    if any(t.type_id!=14 or t.shape[0]!=2048 or t.shape[1] not in (4096,8192) for t in tensors):
        parser.error('Expected original Q6_K projections of width2048')
    args.output.mkdir(exist_ok=False)
    with args.model.open('rb') as source:
        for tensor in tensors:
            source.seek(header.data_start+tensor.offset)
            raw=source.read(tensor.expected_bytes())
            if len(raw)!=tensor.expected_bytes(): raise ValueError('Truncated model tensor')
            path=args.output/(tensor.name+'.bin')
            with path.open('xb') as output:
                output.write(struct.pack('<ii',tensor.shape[1],tensor.shape[0]))
                output.write(raw)
            print(path)

if __name__=='__main__': main()
