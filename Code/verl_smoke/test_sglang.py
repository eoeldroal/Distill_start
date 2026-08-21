"""sglang 단독 기동 확인 (Qwen3-1.7B-Base, canonical sampler).
sglang은 multiprocessing spawn을 쓰므로 __main__ 가드가 필수다."""
import sglang as sgl

def main():
    llm = sgl.Engine(model_path="Qwen/Qwen3-1.7B-Base", dtype="bfloat16",
                     mem_fraction_static=0.55, log_level="error")
    prompts = ["Problem: If x+1/x=5, compute x^3+1/x^3.\nSolution: Let's solve this step by step.\n"]

    # 1) canonical sampler (truncation 없는 full softmax)로 생성
    out = llm.generate(prompts, {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 80})
    print("=== 생성 결과 ===")
    print(out[0]["text"][:400])
    print("=== meta ===")
    print({k: v for k, v in out[0]["meta_info"].items()
           if k in ("completion_tokens", "finish_reason")})

    # 2) verl의 teacher 채점 경로: prompt_logprobs (top-k)
    out2 = llm.generate(prompts, {"temperature": 1.0, "max_new_tokens": 1},
                        return_logprob=True, logprob_start_len=0, top_logprobs_num=8)
    mi = out2[0]["meta_info"]
    itl = mi.get("input_token_logprobs") or []
    itop = mi.get("input_top_logprobs") or []
    print("=== prompt_logprobs (verl teacher 경로) ===")
    print("input_token_logprobs 길이:", len(itl))
    print("input_top_logprobs 길이:", len(itop))
    if len(itop) > 1 and itop[1]:
        print("위치1의 top-3:", itop[1][:3])
    llm.shutdown()
    print("\nsglang Engine 정상 동작")

if __name__ == "__main__":
    main()
