"""Independent NumPy Q8 inputs and Q4_K FP64 affine references.

Consumes a SHA-bound bounded GGUF/gguf.quants fixture from the separate FP32
candidate; it never reads/changes a model or executes a native/GPU process.
Both the original FP32 dots and the changed Q8-input dots are stored separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np

OLD_MAGIC=0x31324634514D464C
MAGIC=0x31513444504D464C
IDS=(0,63,17,42)
ROUTES=(IDS,(63,42,0,17),(17,0,42,63))

def read(stream,fmt):
    value=stream.read(struct.calcsize(fmt))
    if len(value)!=struct.calcsize(fmt):raise ValueError('Truncated bounded input')
    return struct.unpack(fmt,value)

def quantize(x):
    groups=x.reshape(x.shape[0],-1,32)
    maximum=np.max(np.abs(groups),axis=-1,keepdims=True)
    ratio=np.divide(groups,maximum,out=np.zeros_like(groups),where=maximum!=0)
    codes=np.clip(np.rint(ratio*np.float32(127)),-127,127).astype('i1').reshape(x.shape)
    scales=(maximum[...,0]/np.float32(127)).astype('<f4')
    decoded=codes.astype(np.float64)*np.repeat(scales.astype(np.float64),32,axis=1)
    return codes,scales,decoded

def emit(path,cases,magic=MAGIC):
    with path.open('xb') as output:
        output.write(struct.pack('<QI',magic,len(cases)))
        for rows,width,experts,name,raw,weights,runs in cases:
            output.write(struct.pack('<IIIIQI',rows,width,experts,len(runs),len(raw),len(name)))
            output.write(name);output.write(raw);output.write(weights.astype('<f4').tobytes())
            for tokens,separate,routing,ninputs,x,codes,scales,ids,dots,original in runs:
                output.write(struct.pack('<IIII',tokens,separate,routing,ninputs))
                for array in (x,codes,scales,ids,dots,original):output.write(array.tobytes())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,type=Path);ap.add_argument('--expected-sha256',required=True)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--prefill',action='store_true',help='T17/63/104/128 expert-only references, one bounded file per tensor')
    args=ap.parse_args()
    if args.input.stat().st_size>64*1024**2 or hashlib.sha256(args.input.read_bytes()).hexdigest()!=args.expected_sha256:
        raise ValueError('Unbound/oversized source fixture')
    if args.output.exists() or args.output.with_suffix('.json').exists():raise ValueError('Use new output paths')
    all_cases=[]
    with args.input.open('rb') as source:
        magic,count=read(source,'<QI')
        if magic!=OLD_MAGIC or not 1<=count<=16:raise ValueError('Expected bounded independent Q4 fixture')
        for _ in range(count):
            rows,width,experts,inputs,rawbytes,namelen=read(source,'<IIIIQI')
            assert rows==17 and width%256==0 and experts in (1,64) and inputs==4 and namelen<161
            assert rawbytes==experts*rows*(width//256)*144
            name=source.read(namelen);raw=source.read(rawbytes)
            x0=np.frombuffer(source.read(4*width*4),dtype='<f4').reshape(4,width).copy()
            weights=np.frombuffer(source.read((1 if experts==1 else 4)*rows*width*4),dtype='<f4').reshape(-1,rows,width).copy()
            source.read((4 if experts==1 else 24)*rows*8)  # Old full-FP32 dots are not the Q8 oracle.
            if args.prefill and experts!=64:continue
            runs=[]
            for tokens in ((17,63,104,128) if args.prefill else ((1,2,3,4,5,8) if experts==1 else (1,2,3,4,8))):
                for separate in ((False,) if experts==1 else (False,True)):
                    for routing in (range(1) if experts==1 else range(3)):
                        ninputs=tokens*(4 if experts==64 and separate else 1)
                        x=np.stack([x0[(i+routing)%4]*np.float32(1+.03125*i) for i in range(ninputs)]).astype('<f4')
                        x[:,0:32]=0
                        x[:,32:64]=np.arange(-16,16,dtype='<f4')
                        x[:,64:96]=np.array([-127,127,.5,1.5,-.5,-1.5,2.5,-2.5]*4,dtype='<f4')
                        x[:,96:128]=np.float32(.03125);x[:,128:160]=np.float32(-.03125)
                        x[:,160:192]*=np.float32(1e-30)
                        codes,scales,qx=quantize(x)
                        ids=np.array([ROUTES[(routing+t)%3] for t in range(tokens)],dtype='<i4') if experts==64 else np.zeros(0,dtype='<i4')
                        reference=[];original=[]
                        for token in range(tokens):
                            for pick in range(4 if experts==64 else 1):
                                slot=token*4+pick;source_row=slot if separate else token
                                matrix=weights[IDS.index(int(ids[token,pick]))] if experts==64 else weights[0]
                                reference.extend(matrix.astype(np.float64)@qx[source_row])
                                original.extend(matrix.astype(np.float64)@x[source_row].astype(np.float64))
                        runs.append((tokens,int(separate),routing,ninputs,x,codes,scales,ids,np.asarray(reference,dtype='<f8'),np.asarray(original,dtype='<f8')))
            all_cases.append((rows,width,experts,name,raw,weights,runs))
        if source.read(1):raise ValueError('Trailing bounded input')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.prefill:
        args.output.mkdir(exist_ok=False);records=[]
        for index,case in enumerate(all_cases):
            path=args.output/('tensor-'+str(index)+'.bin');emit(path,[case],0x32513444504D464C)
            if path.stat().st_size>64*1024**2:raise ValueError('Prefill tensor reference exceeds bounded64MiB file')
            records.append({'file':path.name,'tensor':case[3].decode('utf-8'),'runs':len(case[-1]),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt={'schema':1,'source_fixture_sha256':args.expected_sha256,'tokens':[17,63,104,128],
                 'cases':len(all_cases),'runs':sum(len(case[-1]) for case in all_cases),'files':records,
                 'selected_experts':IDS,'routes':'three permutations of the four genuine extracted experts; occupancy beyond these four is not claimed',
                 'quantizer_reference':'independent NumPy FP32 max32/div127/rint; actual GPU quantizer verified separately',
                 'dot_reference':'NumPy decoded GGUF Q4_K weights dot Q8 inputs in FP64; original FP32 dots stored separately',
                 'activation_precision_changed':True,'new_source_model_bytes_read':0,'native_or_gpu_or_model_started':False}
        (args.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(receipt,indent=2));return
    emit(args.output,all_cases)
    receipt={'schema':1,'cases':len(all_cases),'runs':sum(len(c[-1]) for c in all_cases),
             'source_fixture_sha256':args.expected_sha256,'fixture_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
             'input_reference':'independent NumPy float32 max32/div127/rint ties-to-even/int8; stored codes and FP32 scales',
             'dot_reference':'independent gguf NumPy Q4_K float rows dot (int8_code * float_scale) in FP64',
             'original_fp32_dots_are_separate':True,'activation_quantization_changes_precision':True,
             'new_source_model_bytes_read':0,'native_or_gpu_or_model_started':False}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
