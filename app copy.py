import time
import uuid
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os, requests
import pydicom
import nibabel as nib
import numpy as np
import requests
from werkzeug.utils import secure_filename
from requests_toolbelt.multipart.encoder import MultipartEncoder
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
from flask import send_file
import tempfile
import trimesh
import nrrd
import matplotlib.pyplot as plt
import glob

app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "dicom")

def convert_to_nifti(dicom_dir, output_path):
    slices = []
    for filename in sorted(os.listdir(dicom_dir)):
        if filename.lower().endswith(".dcm"):
            path = os.path.join(dicom_dir, filename)
            try:
                ds = pydicom.dcmread(path)
                slices.append(ds)
            except Exception as e:
                print("읽기 실패:", filename, str(e))
                continue

    if not slices:
        raise Exception("DICOM 파일이 없습니다.")

    # z축 정렬
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 픽셀 데이터 스택
    image_3d = np.stack([s.pixel_array for s in slices], axis=-1)

    # spacing 계산 (z spacing 안전 처리 포함)
    try:
        z_spacing = float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        if z_spacing == 0.0:
            print("⚠️ z-spacing이 0입니다. 기본값 1.0 사용")
            z_spacing = 1.0
    except Exception as e:
        print("⚠️ z-spacing 계산 실패, 기본값 1.0 사용:", str(e))
        z_spacing = 1.0

    try:
        spacing = [
            float(slices[0].PixelSpacing[0]),
            float(slices[0].PixelSpacing[1]),
            z_spacing
        ]
    except Exception as e:
        print("⚠️ PixelSpacing 가져오기 실패, 기본값 [1.0, 1.0, 1.0] 사용:", str(e))
        spacing = [1.0, 1.0, z_spacing]

    affine = np.diag(spacing + [1.0])
    nifti_img = nib.Nifti1Image(image_3d, affine)
    nib.save(nifti_img, output_path)
    print("NIfTI 저장 완료:", output_path)

@app.route("/upload-dicom", methods=["POST"])
def upload_dicom():
    folder_name = request.form.get("folder", "default")
    target_dir = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    files = request.files.getlist("dicomFiles")
    if not files:
        return jsonify({"success": False, "message": "No files uploaded"}), 400

    file_names = []
    for file in files:
        filename_only = os.path.basename(file.filename)
        save_path = os.path.join(target_dir, filename_only) 
        file.save(save_path)
        file_names.append(filename_only)

    # niivue-manifest 생성
    manifest_path = os.path.join(target_dir, "niivue-manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(file_names))

    manifest_url = f"http://127.0.0.1:5000/uploads/dicom/{folder_name}/niivue-manifest.txt"

    # nii 파일 변환 및 저장
    nii_path = os.path.join(target_dir, "converted.nii.gz")
    # flipped_path = os.path.join(target_dir, "converted_flipped.nii.gz")

    nii_exists = os.path.exists(nii_path)
    # flipped_exists = os.path.exists(flipped_path)

    try:

        if not nii_exists:
            convert_to_nifti(target_dir, nii_path)
        else:
            print(f"이미 존재: {nii_path}")

        # if not flipped_exists:
        #     flip_nifti_left_right(nii_path, flipped_path)
        # else:
        #     print(f"이미 존재: {flipped_path}")

        original = nib.load(nii_path).get_fdata()
        # flipped = nib.load(flipped_path).get_fdata()
        print(" 왼쪽 끝값 (원본):", original[0, :, :].mean())
        # print(" 왼쪽 끝값 (반전):", flipped[0, :, :].mean())

        nii_url = f"http://127.0.0.1:5000/uploads/dicom/{folder_name}/converted.nii.gz"
        # flipped_url = f"http://127.0.0.1:5000/uploads/dicom/{folder_name}/converted_flipped.nii.gz"
    except Exception as e:
        return jsonify({"success": False, "message": f"NIfTI 변환 실패: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "message": f"{len(files)} files uploaded and converted to NIfTI",
        "manifestUrl": manifest_url,
        "niiUrl": nii_url,
        # "flippedNiiUrl": flipped_url
    })

def flip_nifti_left_right(input_path, output_path):
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.float32)

    flipped_data = np.flip(data, axis=0)

    affine = img.affine.copy()
    affine[0, 0] *= -1
    affine[0, 3] *= -1

    flipped_img = nib.Nifti1Image(flipped_data, affine)
    nib.save(flipped_img, output_path)

    print("좌우 반전된 NIfTI 저장 완료:", output_path)


@app.route("/upload-and-infer", methods=["POST"])
def upload_and_infer():
    folder_name = request.form.get("folder", "infer")
    target_dir = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"요청된 폴더: {folder_name}")
    print(f"저장 경로: {target_dir}")

    files = request.files.getlist("dicomFiles")
    if not files:
        print("업로드된 DICOM 파일 없음")
        return jsonify({"success": False, "message": "No files uploaded"}), 400

    file_names = []
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(target_dir, filename)
        file.save(save_path)
        file_names.append(filename)
        # print(f"저장된 파일: {filename}")
        inject_phase_info(save_path, phase='PV')

    # inspect_dicom_tags("/Users/kunkioh/workSpace/zerosketch/dicom-server/uploads/dicom/ScalarVolume")
    # phase_guess = guess_phase_from_dicom_folder(target_dir)
    # print(f"📌 추정된 위상: {phase_guess}")

    nii_path = os.path.join(target_dir, "converted.nii.gz")
    nrrd_path = os.path.join(target_dir, "inferred.nrrd")

    # NIfTI 변환
    if not os.path.exists(nii_path):
        try:
            print("DICOM → NIfTI 변환 시작")
            convert_to_nifti(target_dir, nii_path)
            print("NIfTI 변환 완료:", nii_path)
        except Exception as e:
            print("NIfTI 변환 실패:", str(e))
            return jsonify({"success": False, "message": f"NIfTI 변환 실패: {str(e)}"}), 500
    else:
        print("이미 변환된 NIfTI 파일이 존재함:", nii_path)

    size_mb = os.path.getsize(nii_path) / (1024 * 1024)
    print(f"NIfTI 파일 크기: {size_mb:.2f} MB")
    # SMC 추론 요청
    if not os.path.exists(nrrd_path):
        try:
            print("SMC 추론 서버로 전송 중...")
            res = send_large_file_with_progress(nii_path)
            print("SMC 응답 코드:", res.status_code)

            if res.status_code != 200:
                print("SMC 서버 오류 응답:", res.text)
                return jsonify({"success": False, "message": f"SMC 서버 에러: {res.text}"}), 500

            with open(nrrd_path, "wb") as out_f:
                out_f.write(res.content)
            print("NRRD 저장 완료:", nrrd_path)
        except requests.exceptions.ReadTimeout:
            print("SMC 응답 지연으로 timeout 발생")
            return jsonify({"success": False, "message": "SMC 응답 지연(timeout)"}), 504

        except requests.exceptions.RequestException as e:
            print("SMC 요청 중 에러:", str(e))
            return jsonify({"success": False, "message": f"SMC 전송 실패: {str(e)}"}), 500
    else:
        print("이미 존재하는 NRRD 파일 사용:", nrrd_path)

    nii_url = f"http://127.0.0.1:5000/uploads/dicom/{folder_name}/converted.nii.gz"
    nrrd_url = f"http://127.0.0.1:5000/uploads/dicom/{folder_name}/inferred.nrrd"

    print("모든 처리 완료, 결과 반환")

    return jsonify({
        "success": True,
        "message": "DICOM → NIfTI 변환 및 AI 추론 처리 완료",
        "niiUrl": nii_url,
        "nrrdUrl": nrrd_url
    })

def send_large_file_with_progress(nii_path):
    file_size = os.path.getsize(nii_path)
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading")

    def callback(monitor):
        progress.update(monitor.bytes_read - callback.last_bytes)
        callback.last_bytes = monitor.bytes_read
    callback.last_bytes = 0

    encoder = MultipartEncoder(
        fields={"file": ("converted.nii.gz", open(nii_path, "rb"), "application/octet-stream")}
        # fields={"file": ("phase.nii.gz", open(nii_path, "rb"), "application/octet-stream")}
    )

    monitor = MultipartEncoderMonitor(encoder, callback)

    headers = {"Content-Type": monitor.content_type}

    response = requests.post(
        # "https://smc-ssiso-ai.ngrok.app/nifti-to-nrrd",
        "https://smc-ssiso-ai.ngrok.app/infer/hcc-pv/?output_format=.nrrd",
        data=monitor,
        headers=headers,
        timeout=(30, 300),
    )

    progress.close()
    return response

def guess_phase_from_dicom_folder(dicom_dir):
    phases = []

    for filename in sorted(os.listdir(dicom_dir)):
        if filename.lower().endswith('.dcm'):
            path = os.path.join(dicom_dir, filename)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)

                description = ""
                if 'SeriesDescription' in ds:
                    description += ds.SeriesDescription.lower() + " "
                if 'ProtocolName' in ds:
                    description += ds.ProtocolName.lower()

                if 'pv' in description or 'portal' in description:
                    phases.append('PV')
                elif 'ap' in description or 'arterial' in description:
                    phases.append('AP')
                elif 'vp' in description or 'venous' in description:
                    phases.append('VP')
                elif 'dp' in description or 'delay' in description:
                    phases.append('DP')
                elif 'non-contrast' in description or 'nc' in description:
                    phases.append('NC')
                else:
                    phases.append('Unknown')
            except Exception as e:
                print("❌ 파일 처리 실패:", filename, str(e))
                continue

    # 위상별 개수 카운트
    from collections import Counter
    counter = Counter(phases)
    print("\n📊 위상 분포:")
    for phase, count in counter.items():
        print(f"  {phase}: {count}개")

    # 가장 많은 위상을 추정 결과로 반환
    if counter:
        main_phase = counter.most_common(1)[0][0]
        print(f"\n✅ 가장 추정되는 위상: {main_phase}")
        return main_phase
    else:
        print("❗ 위상 정보를 판단할 수 없습니다.")
        return "Unknown"

def inspect_dicom_tags(folder):
    for file in sorted(os.listdir(folder)):
        if file.endswith('.dcm'):
            path = os.path.join(folder, file)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                print(f"\n📄 {file}")
                print("  SeriesDescription:", getattr(ds, 'SeriesDescription', '없음'))
                print("  ProtocolName     :", getattr(ds, 'ProtocolName', '없음'))
            except Exception as e:
                print(f"❌ 읽기 실패: {file} - {e}")
            break  # 첫 파일만 확인


def inject_phase_info(file_path, phase='PV'):
    try:
        ds = pydicom.dcmread(file_path)
        if phase == 'PV':
            ds.SeriesDescription = "Portal Venous Phase"
            ds.ProtocolName = "PortalVenous"
        elif phase == 'AP':
            ds.SeriesDescription = "Arterial Phase"
            ds.ProtocolName = "Arterial"
        elif phase == 'VP':
            ds.SeriesDescription = "Venous Phase"
            ds.ProtocolName = "Venous"
        else:
            ds.SeriesDescription = "Unknown Phase"
            ds.ProtocolName = "Unknown"

        ds.save_as(file_path)
        print(f"✔️ 위상 정보 삽입 완료: {file_path}")
    except Exception as e:
        print(f"❌ 위상 정보 삽입 실패: {file_path} - {e}")     

def parse_slicer_segment_infos(header):
    import re
    segment_infos = {}
    for k, v in header.items():
        m_label = re.match(r"Segment(\d+)_LabelValue$", k)  # ← $ 추가
        m_color = re.match(r"Segment(\d+)_Color$", k)      # ← $ 추가, 정확히 끝에 Color로 끝날 때만
        m_name  = re.match(r"Segment(\d+)_Name$", k)       # ← $ 추가
        if m_label:
            idx = int(m_label.group(1))
            segment_infos.setdefault(idx, {})["label"] = int(str(v).strip())
        if m_color:
            idx = int(m_color.group(1))
            val_str = v if isinstance(v, str) else v.decode() if isinstance(v, bytes) else str(v)
            parts = re.findall(r"-?\d+\.\d+|-?\d+", val_str)
            color_floats = [float(x) for x in parts[:3]] if len(parts) >= 3 else []
            segment_infos.setdefault(idx, {})["color"] = color_floats
        if m_name:
            idx = int(m_name.group(1))
            name_str = v if isinstance(v, str) else v.decode() if isinstance(v, bytes) else str(v)
            segment_infos.setdefault(idx, {})["name"] = name_str
    return segment_infos

def make_obj_and_mtl_for_4d(data, segment_infos, output_dir, folder_name):
    """4D(n채널) nrrd에서 채널별로 obj/mtl 생성"""
    obj_urls = []
    mtl_lines = []
    num_channels = data.shape[0]
    for ch in range(num_channels):
        ch_mask = data[ch]
        info = segment_infos.get(ch)
        name = info.get('name', f'ch{ch+1}') if info else f'ch{ch+1}'

        # 색상 정보 확인
        if not info or "color" not in info:
            print(f"❗채널 {ch+1} ({name}) 컬러 없음, 스킵")
            continue
        color = info["color"]
        if len(color) == 1:
            color = [color[0], 0.0, 0.0]
            print(f"[WARN] 채널 {ch+1} ({name}) 색상 보정: {color}")
        elif len(color) < 3:
            print(f"[WARN] 채널 {ch+1} ({name}) 색상 정보 부족, 스킵")
            continue

        # 3D 마스크인지 확인
        if ch_mask.ndim != 3 or np.sum(ch_mask) == 0:
            print(f"[WARN] 채널 {ch+1} ({name}) 마스크 shape={ch_mask.shape}, skip")
            continue

        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(ch_mask, pitch=3.0)
        obj_name = f"segment_{name}.obj"
        obj_path = os.path.join(output_dir, obj_name)
        mesh.export(obj_path)
        obj_urls.append(f"/uploads/meshes/{folder_name}/{obj_name}")

        mtl_lines.append(f"newmtl segment_{name}")
        mtl_lines.append(f"Kd {color[0]} {color[1]} {color[2]}")
        mtl_lines.append("Ka 0 0 0\n")

        with open(obj_path, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"mtllib segments.mtl\nusemtl segment_{name}\n" + content)
    return obj_urls, mtl_lines


def make_obj_and_mtl_for_3d(data, segment_infos, output_dir, folder_name):
    obj_urls = []
    mtl_lines = []
    label_to_info = {}
    for idx, info in segment_infos.items():
        if "label" in info:
            label_to_info[info["label"]] = info

    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]

    for label_val in unique_labels:
        mask = (data == label_val).astype(np.uint8)
        info = label_to_info.get(label_val)
        if not info or "color" not in info or len(info["color"]) != 3:
            print(f"❗라벨 {label_val} 컬러 없음, 스킵 (info={info})")
            continue
        color = info["color"]

        if mask.ndim != 3 or np.sum(mask) == 0:
            continue

        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(mask, pitch=3.0)
        # === 표면 smoothing 적용 ===
        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=10)
        # =========================

        obj_name = f"segment_{info.get('name', label_val)}.obj"
        obj_path = os.path.join(output_dir, obj_name)
        mesh.export(obj_path)
        obj_urls.append(f"/uploads/meshes/{folder_name}/{obj_name}")

        mtl_lines.append(f"newmtl segment_{info.get('name', label_val)}")
        mtl_lines.append(f"Kd {color[0]} {color[1]} {color[2]}")
        mtl_lines.append("Ka 0 0 0\n")

        with open(obj_path, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"mtllib segments.mtl\nusemtl segment_{info.get('name', label_val)}\n" + content)
    return obj_urls, mtl_lines


@app.route("/nrrd-to-obj", methods=["POST"])
def nrrd_to_obj_api():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "NRRD file missing"}), 400

    uploaded_file = request.files['file']
    folder_name = request.form.get("folder", "meshresult")
    output_dir = os.path.join(BASE_DIR, "uploads", "meshes", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.join(output_dir, "input.nrrd")
    mtl_path = os.path.join(output_dir, "segments.mtl")

    # 1. 이미 mtl/obj 파일이 있으면 패스
    obj_paths = sorted(glob.glob(os.path.join(output_dir, "*.obj")))
    if os.path.exists(mtl_path) and obj_paths:
        # 파일 경로를 URL로 변환
        obj_urls = ["/uploads/meshes/{}/{}".format(folder_name, os.path.basename(p)) for p in obj_paths]
        print(f"✅ 기존 파일을 사용합니다. ({len(obj_urls)}개)")
        return jsonify({
            "success": True,
            "objUrls": obj_urls,
            "mtlUrl": f"/uploads/meshes/{folder_name}/segments.mtl"
        })

    # 2. 새로 저장 및 변환
    uploaded_file.save(input_path)
    print(f"📄 저장된 NRRD: {input_path}")
    data, header = nrrd.read(input_path)

    segment_infos = parse_slicer_segment_infos(header)
    print("=== Slicer 세그먼트 정보 ===")
    for idx, info in sorted(segment_infos.items()):
        print(f"  Segment{idx}: label={info.get('label')}, name={info.get('name')}, color={info.get('color')}")

    obj_urls, mtl_lines = [], []
    if data.ndim == 4:
        print(f"Detected 4D NRRD shape={data.shape}, 각 채널별 처리")
        obj_urls, mtl_lines = make_obj_and_mtl_for_4d(data, segment_infos, output_dir, folder_name)
    elif data.ndim == 3:
        obj_urls, mtl_lines = make_obj_and_mtl_for_3d(data, segment_infos, output_dir, folder_name)
    else:
        print("❗지원하지 않는 NRRD shape:", data.shape)

    # ✅ MTL 저장
    with open(mtl_path, "w") as f:
        f.write("\n".join(mtl_lines))

    print(f"✅ 총 {len(obj_urls)}개 세그먼트 색상 적용 완료")
    for i, obj in enumerate(obj_urls):
        print(f"  ▶ Segment {i+1} URL: {obj}")

    if not obj_urls:
        print("❌ 유효한 세그먼트가 없어 OBJ 파일이 생성되지 않았습니다.")
        return jsonify({"success": False, "message": "No valid segments found"}), 400

    return jsonify({
        "success": True,
        "objUrls": obj_urls,
        "mtlUrl": f"/uploads/meshes/{folder_name}/segments.mtl"
    })

def convert_to_nifti(dicom_dir, output_path):
    slices = []
    for filename in sorted(os.listdir(dicom_dir)):
        if filename.lower().endswith(".dcm"):
            path = os.path.join(dicom_dir, filename)
            try:
                ds = pydicom.dcmread(path)
                slices.append(ds)
            except:
                continue
    if not slices:
        raise Exception("No DICOM files found")
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image_3d = np.stack([s.pixel_array for s in slices], axis=-1)
    try:
        z_spacing = float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        if z_spacing == 0.0:
            z_spacing = 1.0
    except:
        z_spacing = 1.0
    try:
        spacing = [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), z_spacing]
    except:
        spacing = [1.0, 1.0, z_spacing]
    affine = np.diag(spacing + [1.0])
    nifti_img = nib.Nifti1Image(image_3d, affine)
    nib.save(nifti_img, output_path)


def send_to_smc(nii_path):
    with open(nii_path, "rb") as f:
        res = requests.post(
            "https://smc-ssiso-ai.ngrok.app/infer/hcc-pv/?output_format=.nrrd",
            files={"file": ("converted.nii.gz", f, "application/octet-stream")},
            timeout=(30, 300)
        )
    return res


def parse_slicer_segment_infos(header):
    import re
    segment_infos = {}
    for k, v in header.items():
        m_label = re.match(r"Segment(\d+)_LabelValue$", k)
        m_color = re.match(r"Segment(\d+)_Color$", k)
        m_name = re.match(r"Segment(\d+)_Name$", k)
        if m_label:
            idx = int(m_label.group(1))
            segment_infos.setdefault(idx, {})["label"] = int(str(v).strip())
        if m_color:
            idx = int(m_color.group(1))
            parts = list(map(float, v.strip().split()))
            segment_infos.setdefault(idx, {})["color"] = parts[:3]
        if m_name:
            idx = int(m_name.group(1))
            segment_infos.setdefault(idx, {})["name"] = str(v).strip()
    return segment_infos


def make_obj_and_mtl(data, segment_infos, output_dir, folder_name):
    obj_urls, mtl_lines = [], []
    label_to_info = {v['label']: v for v in segment_infos.values() if 'label' in v}
    for label_val in np.unique(data):
        if label_val == 0:
            continue
        mask = (data == label_val).astype(np.uint8)
        info = label_to_info.get(label_val)
        if not info or "color" not in info:
            continue
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(mask, pitch=3.0)
        obj_name = f"segment_{info.get('name', label_val)}.obj"
        obj_path = os.path.join(output_dir, obj_name)
        mesh.export(obj_path)
        obj_urls.append(f"/uploads/dicom/{folder_name}/{obj_name}")
        mtl_lines.append(f"newmtl segment_{info.get('name', label_val)}")
        mtl_lines.append(f"Kd {info['color'][0]} {info['color'][1]} {info['color'][2]}")
        mtl_lines.append("Ka 0 0 0\n")
        with open(obj_path, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"mtllib segments.mtl\nusemtl segment_{info.get('name', label_val)}\n" + content)
    return obj_urls, mtl_lines

@app.route("/upload-and-infer-all", methods=["POST"])
def upload_and_infer_all():
    t_start = time.time()
    folder_name = str(uuid.uuid4())[:8]
    target_dir = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"[1/6] 폴더 생성 완료: {target_dir}")

    # DICOM 업로드
    files = request.files.getlist("dicomFiles")
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(target_dir, filename)
        file.save(save_path)
    print(f"[2/6] DICOM {len(files)}개 저장 완료")

    # NIfTI 변환
    nii_path = os.path.join(target_dir, "converted.nii.gz")
    if not os.path.exists(nii_path):
        t1 = time.time()
        convert_to_nifti(target_dir, nii_path)
        print(f"[3/6] NIfTI 변환 완료 ({round(time.time() - t1, 2)}초)")
    else:
        print(f"[3/6] NIfTI 이미 존재: {nii_path}")

    # 추론 (SMC 서버 호출)
    nrrd_path = os.path.join(target_dir, "inferred.nrrd")
    if not os.path.exists(nrrd_path):
        t2 = time.time()
        res = send_to_smc(nii_path)
        if res.status_code != 200:
            print("[오류] SMC 추론 실패")
            return jsonify({"success": False, "message": "SMC server error"}), 500
        with open(nrrd_path, "wb") as f:
            f.write(res.content)
        print(f"[4/6] SMC 추론 완료 ({round(time.time() - t2, 2)}초)")
    else:
        print(f"[4/6] NRRD 이미 존재: {nrrd_path}")

    # NRRD → OBJ/MTL 변환
    data, header = nrrd.read(nrrd_path)
    segment_infos = parse_slicer_segment_infos(header)

    t3 = time.time()
    obj_urls, mtl_lines = make_obj_and_mtl(data, segment_infos, target_dir, folder_name)
    print(f"[5/6] OBJ 세그먼트 {len(obj_urls)}개 생성 완료 ({round(time.time() - t3, 2)}초)")

    # MTL 저장
    mtl_path = os.path.join(target_dir, "segments.mtl")
    with open(mtl_path, "w") as f:
        f.write("\n".join(mtl_lines))
    print(f"[6/6] MTL 저장 완료: {mtl_path}")

    total_sec = round(time.time() - t_start, 2)
    print(f"[완료] 전체 처리 시간: {total_sec}초")

    return jsonify({
        "success": True,
        "niiUrl": f"/uploads/dicom/{folder_name}/converted.nii.gz",
        "nrrdUrl": f"/uploads/dicom/{folder_name}/inferred.nrrd",
        "objUrls": obj_urls,
        "mtlUrl": f"/uploads/dicom/{folder_name}/segments.mtl"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)