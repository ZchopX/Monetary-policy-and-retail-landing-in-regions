import contextlib,io,json,sys,hashlib,inspect
import numpy as np,pandas as pd
from systemgmmkit.pydynpd_backend import _apply_numpy_compatibility_shims
from pydynpd import regression,specification_tests
data=pd.read_csv(sys.argv[1]); spec=json.load(open(sys.argv[2],encoding="utf8")); out=io.StringIO()
try:
 _apply_numpy_compatibility_shims()
 with contextlib.redirect_stdout(out): fitted=regression.abond(spec["pydynpd_command"],data,["Region","Date"])
 model=fitted.models[0]; info=model.z_information; height=int(model.z_list.shape[0]/model.N); z=np.concatenate([model.z_list[i*height:(i+1)*height,:] for i in range(model.N)],axis=1).T; np.savetxt(sys.argv[4],z,delimiter=","); s=np.linalg.svd(z,compute_uv=False); tol=float(np.finfo(float).eps*max(z.shape)*s[0]) if len(s) else 0.; ar={x.lag:x.P_value for x in model.AR_list}; h=model.hansen
 if z.shape[1]!=len(spec["instrument_labels"]): raise ValueError("Python instrument labels do not match canonical specification")
 result={"status":"completed","raw_console":out.getvalue(),"groups":int(model.N),"instrument_count":int(info.num_instr),"instrument_rank":int((s>tol).sum()),"singular_values":s.tolist(),"instrument_labels":spec["instrument_labels"],"key_sha256":spec["key_sha256"],"input_sha256":spec["input_sha256"],"ar1_p":ar.get(1),"ar2_p":ar.get(2),"hansen_statistic":getattr(h,"test_value",None),"hansen_df":getattr(h,"df",None),"hansen_p":getattr(h,"p_value",None),"hansen_robustness":"two_step_empirical_moment_covariance","hansen_source_sha256":hashlib.sha256(inspect.getsource(specification_tests.hansen_overid).encode()).hexdigest()}
except BaseException as e: result={"status":"failed","raw_console":out.getvalue(),"error_type":type(e).__name__,"error":str(e)}
json.dump(result,open(sys.argv[3],"w",encoding="utf8"),ensure_ascii=False)