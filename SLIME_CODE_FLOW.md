# slime 코드 흐름 가이드

> LLM을 slime으로 강화학습시킬 때의 함수 흐름과 각 함수의 역할 설명

---

## 전체 흐름 개요

```mermaid
sequenceDiagram
    participant Main as train.py
    participant RM as RolloutManager
    participant DS as DataSource
    participant SG as SGLang Engine
    participant RMHub as Reward Model Hub
    participant Actor as Actor Model
    
    Main->>RM: create_rollout_manager()
    Main->>Actor: create_training_models()
    Main->>Actor: update_weights() [초기]
    
    loop 각 Rollout
        Main->>RM: generate(rollout_id)
        RM->>DS: get_samples()
        DS-->>RM: [Sample(prompt, label), ...]
        
        RM->>SG: generate_rollout_async()
        loop 각 샘플 그룹
            SG->>SG: generate() [SGLang HTTP 요청]
            SG-->>SG: response, tokens
            SG->>RMHub: async_rm(sample)
            RMHub-->>SG: reward
        end
        SG-->>RM: [Sample(..., reward), ...]
        
        RM->>RM: _convert_samples_to_train_data()
        RM-->>Main: rollout_data_ref
        
        Main->>Actor: async_train(rollout_id, data_ref)
        Actor->>Actor: compute_log_prob()
        Actor->>Actor: compute_advantages()
        Actor->>Actor: policy_loss_function()
        Actor->>Actor: backward() & step()
        Actor-->>Main: 완료
        
        Main->>Actor: update_weights()
        Actor->>RM: 새 가중치 전송
        RM->>SG: 가중치 로드
    end
```

---

## 1. 초기화 단계

### `train.py::train()`
**역할**: 메인 훈련 루프의 진입점

**주요 작업**:
- GPU 리소스 할당 (`create_placement_groups`)
- RolloutManager 생성 및 SGLang 엔진 초기화
- Actor/Critic 모델 생성 및 초기화
- 초기 가중치 동기화

**호출 흐름**:
```
train()
  → create_placement_groups()      # GPU 리소스 분배
  → create_rollout_manager()       # Rollout 시스템 초기화
  → create_training_models()       # 학습 모델 초기화
  → actor_model.update_weights()  # 초기 가중치 동기화
```

---

### `placement_group.py::create_rollout_manager()`
**역할**: RolloutManager Ray Actor 생성

**주요 작업**:
- RolloutManager를 Ray remote actor로 생성
- Rollout 횟수 계산 (num_rollout_per_epoch)
- 가중치 체크 설정 (선택적)

**반환값**: `(rollout_manager, num_rollout_per_epoch)`

---

### `rollout.py::RolloutManager.__init__()`
**역할**: Rollout 시스템 초기화

**주요 작업**:
- SGLang Router 시작 (`_start_router`)
- DataSource 로드 (프롬프트 데이터 관리)
- Rollout 함수 로드 (`generate_rollout`, `eval_rollout`)
- SGLang Engine 초기화 (`init_rollout_engines`)
- Health Monitor 시작 (선택적)

**핵심 속성**:
- `self.data_source`: 프롬프트 데이터 제공
- `self.generate_rollout`: 실제 생성 함수
- `self.all_rollout_engines`: SGLang 엔진 리스트

---

## 2. Rollout 생성 단계

### `train.py::train()` - 메인 루프
**역할**: Rollout → Training → Weight Update 반복

**핵심 루프**:
```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # 1. Rollout: 데이터 생성
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    
    # 2. Training: 모델 학습
    actor_model.async_train(rollout_id, rollout_data_ref)
    
    # 3. Weight Update: Rollout 엔진에 새 가중치 전송
    actor_model.update_weights()
```

---

### `rollout.py::RolloutManager.generate()`
**역할**: 하나의 rollout 생성 및 훈련 데이터 변환

**주요 작업**:
1. Rollout 데이터 생성 (`_get_rollout_data`)
2. 디버그 데이터 저장 (선택적)
3. Sample을 훈련 데이터로 변환 (`_convert_samples_to_train_data`)
4. Data Parallel로 분할 (`_split_train_data_by_dp`)

**반환값**: Data Parallel별로 분할된 훈련 데이터 (Ray ObjectRef)

**호출 흐름**:
```
generate(rollout_id)
  → _get_rollout_data()
    → call_rollout_fn(generate_rollout, ...)
      → generate_rollout_async() [sglang_rollout.py]
  → _convert_samples_to_train_data()
  → _split_train_data_by_dp()
```

---

### `rollout.py::RolloutManager._get_rollout_data()`
**역할**: Rollout 함수 호출하여 Sample 리스트 생성

**주요 작업**:
- `generate_rollout` 함수 호출
- 중첩 리스트 평탄화
- 샘플 수 트리밍 (global_batch_size에 맞춤)

**반환값**: `(samples: list[Sample], metrics: dict)`

---

### `sglang_rollout.py::generate_rollout_async()`
**역할**: 실제 텍스트 생성 및 리워드 계산

**주요 작업**:
1. DataSource에서 프롬프트 샘플 가져오기
2. 생성 태스크 제출 (`submit_generate_tasks`)
3. 완료된 태스크 수집 및 동적 필터링
4. 목표 샘플 수 달성까지 반복

**핵심 로직**:
```python
while len(data) < target_data_size:
    samples = data_source.get_samples(num_samples)
    state.submit_generate_tasks(samples)  # 비동기 생성 시작
    
    done, pendings = await asyncio.wait(pendings, ...)
    for task in done:
        group = task.result()  # [Sample, ...]
        if dynamic_filter.keep(group):
            data.append(group)
```

**반환값**: `RolloutFnTrainOutput(samples=data, metrics=...)`

---

### `sglang_rollout.py::generate_and_rm_group()`
**역할**: 샘플 그룹 단위로 생성 및 리워드 계산

**주요 작업**:
- 그룹 내 각 샘플에 대해 `generate_and_rm` 병렬 실행
- 그룹 리워드 계산 (group_rm인 경우)

**호출 흐름**:
```
generate_and_rm_group(group)
  → generate_and_rm(sample1) [병렬]
  → generate_and_rm(sample2) [병렬]
  → ...
  → batched_async_rm(group) [group_rm인 경우]
```

---

### `sglang_rollout.py::generate_and_rm()`
**역할**: 개별 샘플 생성 + 리워드 계산

**주요 작업**:
1. SGLang으로 텍스트 생성 (`generate`)
2. 리워드 계산 (`async_rm`)

**호출 흐름**:
```
generate_and_rm(sample, sampling_params)
  → generate(args, sample, sampling_params)
    → SGLang Router HTTP POST
    → sample.response, sample.tokens 업데이트
  → async_rm(args, sample)
    → sample.reward 업데이트
```

---

### `sglang_rollout.py::generate()`
**역할**: SGLang Router를 통해 텍스트 생성

**주요 작업**:
- Tokenizer로 프롬프트 토큰화
- SGLang Router에 HTTP POST 요청
- 생성된 텍스트와 토큰을 Sample에 저장
- Log probability 저장 (PPO용)

**요청 형식**:
```python
payload = {
    "input_ids": prompt_ids,
    "sampling_params": {...},
    "return_logprob": True
}
response = await post(f"http://{router_ip}:{port}/generate", payload)
```

---

### `rm_hub/__init__.py::async_rm()`
**역할**: 리워드 모델로 리워드 계산

**주요 작업**:
- 커스텀 RM 함수가 있으면 호출 (`custom_rm_path`)
- 아니면 `rm_type`에 따라 rule-based RM 사용
  - `math`: 수학 문제 정답 검증
  - `f1`: F1 스코어 계산
  - `gpqa`: GPQA 리워드
  - `remote_rm`: 원격 RM 서버 호출

**반환값**: `float` (리워드 값)

---

### `rollout.py::RolloutManager._convert_samples_to_train_data()`
**역할**: Sample 리스트를 훈련 데이터 딕셔너리로 변환

**주요 작업**:
- 리워드 후처리 (정규화 등)
- 토큰, 리워드, loss_mask 등 추출
- Rollout log probs 포함 (PPO용)

**반환 데이터 구조**:
```python
{
    "tokens": [[token_ids], ...],
    "response_lengths": [int, ...],
    "rewards": [float, ...],
    "raw_reward": [float, ...],
    "loss_masks": [[0/1], ...],
    "rollout_log_probs": [[float], ...],
    "truncated": [0/1, ...],
    ...
}
```

---

## 3. 훈련 단계

### `actor_group.py::RayTrainGroup.async_train()`
**역할**: 모든 Actor에 훈련 태스크 분배

**주요 작업**:
- 각 Actor에 `train()` 메서드 비동기 호출
- Data Parallel로 데이터 분할

**반환값**: Ray ObjectRef 리스트

---

### `megatron_utils/actor.py::MegatronTrainRayActor.train()`
**역할**: 하나의 rollout에 대한 훈련 실행

**주요 작업**:
1. Rollout 데이터 로드 및 전처리
2. Actor 또는 Critic 훈련 분기

**호출 흐름**:
```
train(rollout_id, rollout_data_ref)
  → get_rollout_data()           # 데이터 로드
  → train_actor() or train_critic()
```

---

### `megatron_utils/actor.py::MegatronTrainRayActor.train_actor()`
**역할**: Actor 모델 훈련 (PPO)

**주요 작업**:
1. 현재 정책의 log prob 계산 (`compute_log_prob`)
2. Advantage 계산 (`compute_advantages_and_returns`)
3. Policy Loss 계산 및 학습 (`train`)

**호출 흐름**:
```
train_actor(rollout_id, rollout_data)
  → compute_log_prob()              # Forward pass
  → compute_advantages_and_returns() # Advantage 계산
  → train()                         # Loss 계산 & Backward
    → policy_loss_function()
```

---

### `training_utils/loss.py::policy_loss_function()`
**역할**: PPO Policy Loss 계산

**주요 작업**:
1. 현재 log prob과 rollout log prob 비교
2. Importance ratio 계산: `r = exp(log_prob - old_log_prob)`
3. PPO clipped loss: `min(r * A, clip(r, 1-ε, 1+ε) * A)`
4. Entropy loss 추가
5. KL loss 추가 (선택적)

**Loss 구성**:
```
loss = pg_loss - entropy_coef * entropy_loss + kl_coef * kl_loss
```

---

## 4. 가중치 동기화

### `actor_group.py::RayTrainGroup.update_weights()`
**역할**: Actor 모델의 가중치를 Rollout 엔진에 전송

**주요 작업**:
- Rank 0의 가중치를 다른 rank로 브로드캐스트
- RolloutManager에 새 가중치 전송
- SGLang Engine에 가중치 로드

**호출 흐름**:
```
update_weights()
  → actor.update_weights() [각 rank]
  → rollout_manager.update_weights() [RolloutManager]
  → engine.load_weights() [SGLang Engine]
```

---

## 핵심 함수 체인 요약

### Rollout 생성 체인
```
train() 
  → RolloutManager.generate()
    → _get_rollout_data()
      → generate_rollout_async()
        → generate_and_rm_group()
          → generate_and_rm()
            → generate() [SGLang]
            → async_rm() [Reward]
    → _convert_samples_to_train_data()
```

### 훈련 체인
```
train()
  → RayTrainGroup.async_train()
    → TrainRayActor.train()
      → train_actor()
        → compute_log_prob()
        → compute_advantages_and_returns()
        → policy_loss_function()
        → backward() & optimizer.step()
```

### 가중치 동기화 체인
```
train()
  → RayTrainGroup.update_weights()
    → TrainRayActor.update_weights()
      → RolloutManager.update_weights()
        → SGLangEngine.load_weights()
```

---

## 주요 데이터 구조

### `Sample` (`utils/types.py`)
**역할**: 하나의 생성 샘플을 나타내는 데이터 클래스

**주요 속성**:
- `prompt`: 입력 프롬프트
- `response`: 생성된 응답
- `tokens`: 토큰 ID 리스트
- `reward`: 리워드 값
- `rollout_log_probs`: Rollout 시의 log probability
- `loss_mask`: Loss를 적용할 토큰 마스크
- `status`: PENDING, COMPLETED, TRUNCATED, ABORTED

### `RolloutBatch` (`utils/types.py`)
**역할**: 훈련에 사용되는 배치 데이터

**주요 속성**:
- `tokens`: 토큰 리스트
- `rewards`: 리워드 리스트
- `advantages`: Advantage 리스트
- `loss_masks`: Loss 마스크 리스트
- `rollout_log_probs`: Rollout log probs
- `log_probs`: 현재 정책의 log probs

---

## 참고 자료

- [slime GitHub](https://github.com/THUDM/slime)
- [slime Documentation](https://thudm.github.io/slime/)
- [slime Architecture Blog](https://lmsys.org/blog/2025-07-09-slime/)
