import os
import sys
import time
import uuid
import glob
import json
import gzip
import zipfile
import logging
import tempfile  
from io import BytesIO

# Flask & Web 관련
from flask import Flask, request, jsonify, send_file, send_from_directory, make_response, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# HTTP 요청 관련
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from requests_toolbelt import MultipartEncoderMonitor
from tqdm import tqdm

# 의료 영상 관련
import pydicom
import nibabel as nib
import nrrd

# 데이터 처리 & 과학 계산
import numpy as np
from scipy.ndimage import zoom
from skimage import measure
from skimage.measure import marching_cubes

# 메시 처리
import trimesh
from trimesh.smoothing import filter_taubin

# 시각화 (디버깅용)
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "dicom")



# 로그 설정 (애플리케이션 시작 시 한 번만)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes




@app.route("/", methods=["GET"])
def index():
    return {"status": "running", "message": "Flask server is up!"}




# @app.route('/convert-mesh', methods=['OPTIONS', 'POST'])
# def convert_mesh():
#     import scipy.ndimage as ndi
#     try:
#         label_id = int(request.form.get("label", 1))
#         original_mesh_file = request.files.get("original_mesh")
#         edited_mesh_file = request.files.get("edited_mesh")
#         nrrd_file = request.files.get("nrrd_file")

#         logging.info("===== [convert-mesh 요청 수신] =====")
#         logging.info(f"▶ 라벨 ID: {label_id}")
#         logging.info(f"▶ original_mesh 파일: {original_mesh_file.filename}")
#         logging.info(f"▶ edited_mesh 파일: {edited_mesh_file.filename}")
#         n_bytes = len(nrrd_file.read())
#         logging.info(f"▶ NRRD 파일 크기: {n_bytes} bytes")

#         nrrd_file.seek(0)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".nrrd") as tmp:
#             tmp.write(nrrd_file.read())
#             tmp_path = tmp.name

#         seg_image = sitk.ReadImage(tmp_path)
#         seg_data = sitk.GetArrayFromImage(seg_image)  # (Z, Y, X)
#         size = seg_image.GetSize()  # (X,Y,Z)
#         origin = np.array(seg_image.GetOrigin())
#         spacing = np.array(seg_image.GetSpacing())
#         direction = np.array(seg_image.GetDirection()).reshape(3, 3)
#         inv_direction = np.linalg.inv(direction)

#         logging.info("===== [NRRD 메타데이터] =====")
#         logging.info(f"Shape (Z,Y,X): {seg_data.shape}")
#         logging.info(f"Origin (mm): {origin}")
#         logging.info(f"Spacing: {spacing}")
#         logging.info(f"Direction:\n{direction}")

#         # === 편집 메쉬 로드 (obj) ===
#         edited_mesh = trimesh.load(edited_mesh_file, file_type='obj')
#         verts_edited = np.array(edited_mesh.vertices)
#         faces_edited = np.array(edited_mesh.faces)
#         verts_edited[:, 0] *= -1
#         verts_edited = verts_edited[:, [0, 2, 1]]
#         logging.info(f"[편집 메시] min(mm): {verts_edited.min(0)}, max(mm): {verts_edited.max(0)}, center(mm): {verts_edited.mean(0)}")
#         edited_trimesh = trimesh.Trimesh(vertices=verts_edited, faces=faces_edited, process=True)
#         pitch = float(np.min(spacing)) * 0.8  # 더 촘촘하게

#         # === Voxelize mesh ===
#         vox = edited_trimesh.voxelized(pitch=pitch)
#         vox_matrix = vox.matrix.astype(np.uint8)
#         vox_transform = vox.transform  # 4x4 matrix
#         vox_origin = vox_transform[:3, 3]
#         logging.info(f"[voxelized] matrix shape: {vox_matrix.shape}, origin (from transform): {vox_origin}")
#         logging.info(f"[voxelized] bounding box (mm): min {verts_edited.min(0)}, max {verts_edited.max(0)}")
#         logging.info(f"[NRRD bbox] origin {origin}, end {origin + spacing * (np.array(seg_data.shape)[::-1])}")

#         # === Voxel → NRRD 영역 매핑 ===
#         mask = np.zeros_like(seg_data, dtype=np.uint8)
#         n_filled = 0
#         z_max, y_max, x_max = vox_matrix.shape
#         idx_debug = []
#         for z in range(z_max):
#             for y in range(y_max):
#                 for x in range(x_max):
#                     if vox_matrix[z, y, x] > 0:
#                         pt_vox = np.array([x, y, z, 1])
#                         pt_mm = (vox_transform @ pt_vox)[:3]   # (mm)
#                         rel_mm = pt_mm - origin
#                         idx_xyz = np.dot(inv_direction, rel_mm) / spacing
#                         idx_zyx = np.round(idx_xyz[::-1]).astype(int)
#                         if n_filled < 10:
#                             idx_debug.append((z, y, x, pt_mm.tolist(), idx_zyx.tolist()))
#                         if (
#                             0 <= idx_zyx[0] < mask.shape[0] and
#                             0 <= idx_zyx[1] < mask.shape[1] and
#                             0 <= idx_zyx[2] < mask.shape[2]
#                         ):
#                             mask[idx_zyx[0], idx_zyx[1], idx_zyx[2]] = 1
#                             n_filled += 1
#         logging.info(f"[mask] voxelized mask sum: {mask.sum()}, n_filled: {n_filled}")
#         for i, row in enumerate(idx_debug):
#             logging.info(f"[debug] voxel({row[0]},{row[1]},{row[2]}) mm={row[3]} → nrrd idx={row[4]}")

#         # 내부 채움 및 보정
#         mask_filled = ndi.binary_fill_holes(mask).astype(np.uint8)
#         mask_filled = ndi.binary_closing(mask_filled, iterations=1).astype(np.uint8)  # (optional)
#         logging.info(f"[mask] filled mask sum: {mask_filled.sum()}")

#         # 기존 라벨 삭제 & 새로 할당
#         before_count = np.sum(seg_data == label_id)
#         seg_data[seg_data == label_id] = 0
#         seg_data[mask_filled > 0] = label_id
#         after_count = np.sum(seg_data == label_id)
#         logging.info(f"라벨 {label_id} 교체: 삭제 전 {before_count}, 적용 후 {after_count}")

#         # NRRD 저장
#         new_image = sitk.GetImageFromArray(seg_data)
#         new_image.CopyInformation(seg_image)

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".nrrd") as tmp_out:
#             sitk.WriteImage(new_image, tmp_out.name)
#             tmp_out.flush()
#             tmp_out.seek(0)
#             result_bytes = tmp_out.read()

#         logging.info("===== [변환 완료 → NRRD 반환] =====")
#         return Response(result_bytes, mimetype='application/octet-stream')

#     except Exception as e:
#         logging.exception("❌ convert-mesh 처리 중 오류 발생")
#         return jsonify({"success": False, "message": str(e)}), 500




@app.route('/convert-mesh', methods=['OPTIONS', 'POST'])
def convert_mesh():
    try:
        # === 요청 파라미터 ===
        label_id = int(request.form.get("label", 1))
        original_mesh_file = request.files.get("original_mesh")
        edited_mesh_file = request.files.get("edited_mesh")
        nrrd_file = request.files.get("nrrd_file")

        logging.info("===== [convert-mesh 요청 수신] =====")
        logging.info(f"▶ 라벨 ID: {label_id}")
        logging.info(f"▶ original_mesh 파일: {original_mesh_file.filename}")
        logging.info(f"▶ edited_mesh 파일: {edited_mesh_file.filename}")
        logging.info(f"▶ NRRD 파일 크기: {len(nrrd_file.read())} bytes")

        # === NRRD 로드 ===
        nrrd_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nrrd") as tmp:
            tmp.write(nrrd_file.read())
            tmp_path = tmp.name

        seg_image = sitk.ReadImage(tmp_path)
        seg_data = sitk.GetArrayFromImage(seg_image)  # (Z, Y, X)
        size = seg_image.GetSize()  # (X,Y,Z)
        origin = np.array(seg_image.GetOrigin())
        spacing = np.array(seg_image.GetSpacing())
        direction = np.array(seg_image.GetDirection()).reshape(3, 3)
        inv_direction = np.linalg.inv(direction)

        logging.info("===== [NRRD 메타데이터] =====")
        logging.info(f"Shape (Z,Y,X): {seg_data.shape}")
        logging.info(f"Origin (mm): {origin}")
        logging.info(f"Spacing: {spacing}")
        logging.info(f"Direction:\n{direction}")

        # 라벨 중심 계산 개선
        old_coords = np.argwhere(seg_data == label_id)
        if old_coords.size > 0:
            old_center_voxel = old_coords.mean(axis=0)
            old_center_mm = origin + np.dot(direction, old_center_voxel[::-1] * spacing)
            logging.info(f"기존 라벨 중심(mm): {old_center_mm}")

        # === 원본 메쉬 로드 ===
        original_mesh = trimesh.load(original_mesh_file, file_type='obj')
        verts_original = np.array(original_mesh.vertices)
        verts_original[:, :2] *= -1  # X,Y 반전
        orig_center_mm = verts_original.mean(axis=0)
        logging.info(f"원본 메쉬 중심(mm): {orig_center_mm}")

        # === 편집 메쉬 로드 ===
        edited_mesh = trimesh.load(edited_mesh_file, file_type='obj')
        verts_edited = np.array(edited_mesh.vertices)
        # verts_edited[:, :2] *= -1
        verts_edited[:, 0] *= -1
        verts_edited = verts_edited[:, [0, 2, 1]]
        logging.info(f"편집 메쉬 버텍스 개수: {len(verts_edited)}")
        logging.info(f"편집 메쉬 중심(mm): {verts_edited.mean(axis=0)}")

        # 편집 메시 로드 후
        logging.info(f"[편집 메시] min(mm): {verts_edited.min(0)}, max(mm): {verts_edited.max(0)}, center(mm): {verts_edited.mean(0)}")
        logging.info(f"[편집 메시] 샘플 vertex(mm): {verts_edited[:5].tolist()}")

        # flip 적용 전/후 모두 찍기

        # mm → voxel 변환 전
        logging.info(f"[mm→voxel] 변환 전 샘플(mm): {verts_edited[:5].tolist()}")

        # === 오프셋 적용 ===
        if old_center_mm is not None:
            # 라벨의 bounding box 계산
            label_coords = np.argwhere(seg_data == label_id)
            label_min = label_coords.min(axis=0)
            label_max = label_coords.max(axis=0)

            # 메시의 bounding box 계산 (mm → voxel 변환 후)
            verts_voxel = (inv_direction @ ((verts_edited - origin).T) / spacing[:, None]).T
            verts_voxel_zyx = verts_voxel[:, [2, 1, 0]]
            mesh_min = verts_voxel_zyx.min(axis=0)
            mesh_max = verts_voxel_zyx.max(axis=0)

            # min 좌표를 맞추는 오프셋
            offset_voxel = label_min - mesh_min
            verts_edited += offset_voxel[::-1] * spacing  # ZYX → XYZ

            logging.info(f"적용된 오프셋(mm): {offset_voxel[::-1] * spacing}")

        # === Bounding Box 로그 ===
        logging.info(f"편집 메쉬 BoundingBox(mm): min {verts_edited.min(0)}, max {verts_edited.max(0)}")
        logging.info(f"NRRD BoundingBox(mm): min {origin}, max {origin + spacing * np.array(size)}")

        # === 좌표(mm) → Voxel Index 변환 ===
        logging.info("===== [좌표 변환 → Voxel Index] =====")
        logging.info(f"좌표 변환 이전 샘플(mm): {verts_edited[:5].tolist()}")

        transformed = inv_direction @ ((verts_edited - origin).T)
        indices = (transformed / spacing[:, None]).T  # (N, 3) in XYZ
        indices_int = np.round(indices).astype(int)

        # XYZ → ZYX (NRRD 배열은 Z,Y,X)
        indices_zyx = indices_int[:, [2, 1, 0]]

        # 필터링
        valid_mask = (
            (indices_zyx[:, 0] >= 0) & (indices_zyx[:, 0] < seg_data.shape[0]) &
            (indices_zyx[:, 1] >= 0) & (indices_zyx[:, 1] < seg_data.shape[1]) &
            (indices_zyx[:, 2] >= 0) & (indices_zyx[:, 2] < seg_data.shape[2])
        )
        valid_indices = indices_zyx[valid_mask]

        logging.info(f"mm → voxel float 샘플: {indices[:5].round(2).tolist()}")
        logging.info(f"voxel index 범위(Z,Y,X): min {valid_indices.min(0)}, max {valid_indices.max(0)}")
        logging.info(f"유효 인덱스 개수: {len(valid_indices)}")

        # === 마스크 생성 ===
        mask = np.zeros_like(seg_data, dtype=np.uint8)
        for z, y, x in valid_indices:
            mask[z, y, x] = 1
        mask_filled = binary_fill_holes(mask).astype(np.uint8)

        # === 기존 라벨 삭제 & 새로 할당 ===
        before_count = np.sum(seg_data == label_id)
        seg_data[seg_data == label_id] = 0
        seg_data[mask_filled > 0] = label_id
        after_count = np.sum(seg_data == label_id)
        logging.info(f"라벨 {label_id} 교체: 삭제 전 {before_count}, 적용 후 {after_count}")

        # === NRRD 저장 ===
        new_image = sitk.GetImageFromArray(seg_data)
        new_image.CopyInformation(seg_image)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nrrd") as tmp_out:
            sitk.WriteImage(new_image, tmp_out.name)
            tmp_out.flush()
            tmp_out.seek(0)
            result_bytes = tmp_out.read()

        logging.info("===== [변환 완료 → NRRD 반환] =====")
        return Response(result_bytes, mimetype='application/octet-stream')

    except Exception as e:
        logging.exception("❌ convert-mesh 처리 중 오류 발생")
        return jsonify({"success": False, "message": str(e)}), 500













# @app.route('/convert-mesh', methods=['POST'])
# def convert_mesh():
#     # 1. OBJ 파일 받기
#     mesh_file = request.files.get("mesh_file")
#     if not mesh_file:
#         return {"error": "mesh_file not provided"}, 400

#     # 2. 옵션 파라미터 받기
#     dims = request.form.get("dims")
#     spacing = request.form.get("spacing")
#     dims = [int(x) for x in dims.strip("[]").split(",")]
#     spacing = [float(x) for x in spacing.strip("[]").split(",")]

#     # 3. 메시 로드 (trimesh 사용)
#     mesh = trimesh.load(mesh_file, file_type='obj')

#     # 4. Voxel 변환
#     volume = mesh.voxelized(pitch=spacing[0])  # pitch=voxel 크기
#     voxel_matrix = volume.matrix.astype(np.uint8) * 1  # 1/0 mask

#     # 5. NRRD 변환 (pynrrd)
#     header = {
#         'space': 'left-posterior-superior',
#         'space directions': [[spacing[0],0,0],[0,spacing[1],0],[0,0,spacing[2]]],
#         'kinds': ['domain', 'domain', 'domain']
#     }

#     with BytesIO() as buf:
#         nrrd.write(buf, voxel_matrix, header)
#         buf.seek(0)
#         return Response(buf.read(), mimetype='application/octet-stream')

@app.route('/generate-mesh', methods=['OPTIONS', 'POST'])
def generate_mesh():
    if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
            response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
            return response
    try:
        file = request.files['file']
        file_bytes = file.read()

        with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            data, header = nrrd.read(tmp.name)

        print(header)

        spacing = header.get('space directions', np.eye(3))
        if isinstance(spacing, np.ndarray):
            spacing = np.diag(spacing).tolist()
        else:
            spacing = [5.0, 5.0, 5.0]

        origin = np.array(header.get('space origin', [0, 0, 0]))
        # origin = [0, 0, 0]
        print(f"✅ Spacing: {spacing}")
        print(f"✅ Origin: {origin}")

        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]

        all_verts = []
        label_meshes = []
        scale_factor = 1  # ✅ 업샘플링 배율

        for label in unique_labels:
            mask = (data == label).astype(np.uint8)
            if np.sum(mask) == 0:
                continue

            # ✅ 해상도 업샘플링
            mask_resampled = zoom(mask, scale_factor, order=0)
            spacing_resampled = [s / scale_factor for s in spacing]

            # marching cubes 실행
            verts, faces, _, _ = marching_cubes(mask_resampled, level=0.5, spacing=spacing_resampled)

            # verts: voxel index 좌표
            # spacing_resampled: spacing (mm)
            # origin: NRRD origin (mm)

            # # mm 좌표로 변환
            verts_mm = verts # + origin  # direction이 단위행렬일 때

            # # ✅ 스무딩 적용
            mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces)
            filter_taubin(mesh, lamb=0.4, nu=-0.53, iterations=15)

            verts = mesh.vertices
            faces = mesh.faces

            # ✅ 좌표계 변환
            verts += origin
            # verts[:, 0] *= -1  # L → R
            # verts = verts[:, [0, 1, 2]]

            all_verts.append(verts)
            label_meshes.append({"label": int(label), "verts": verts, "faces": faces})

        # ✅ 중심 맞춤
        all_verts_concat = np.vstack(all_verts)
        center = np.mean(all_verts_concat, axis=0)
        print(f"✅ 전체 메시 중심: {center}")

        meshes = []
        for mesh_data in label_meshes:
            verts = mesh_data["verts"]
            faces = mesh_data["faces"]

            obj_lines = [f"v {v[0]} {v[1]} {v[2]}" for v in verts]
            obj_lines += [f"f {f[0]+1} {f[1]+1} {f[2]+1}" for f in faces]
            obj_data = "\r\n".join(obj_lines)

            meshes.append({"label": mesh_data["label"], "name": f"segment_{mesh_data['label']}", "objData": obj_data})

        return jsonify({"success": True, "meshes": meshes})

    except Exception as e:
        print("❌ 오류 발생:", str(e))
        return jsonify({"success": False, "message": str(e)}), 500








@app.route("/infer-dicom-bundle", methods=["POST"])
def infer_dicom_bundle():
    start_time = time.time()
    files = request.files.getlist("dicomFiles")
    logging.info(f"📥 /infer-dicom-bundle 요청: DICOM 파일 {len(files)}개")

    if not files:
        logging.warning("❌ DICOM 파일이 없음")
        return jsonify({"success": False, "message": "No DICOM files"}), 400

    # 1. DICOM 메모리 로딩
    slices = []
    for i, file in enumerate(files):
        try:
            ds = pydicom.dcmread(file.stream)
            slices.append(ds)
        except Exception as e:
            logging.error(f"❌ DICOM[{i}] 읽기 실패: {str(e)}")
            return jsonify({"success": False, "message": f"DICOM 읽기 실패: {str(e)}"}), 400

    if not slices:
        logging.warning("❌ DICOM slice 없음")
        return jsonify({"success": False, "message": "DICOM slice 없음"}), 400

    # 2. 임시 폴더에 저장
    with tempfile.TemporaryDirectory() as temp_dcm_dir:
        for i, s in enumerate(slices):
            s.save_as(os.path.join(temp_dcm_dir, f"{i:04d}.dcm"))

        # 3. 변환 함수 호출
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmpfile:
            try:
                convert_to_nifti(temp_dcm_dir, tmpfile.name)
                tmpfile.seek(0)
                nii_bytes = tmpfile.read()

                # nii_bytes = convert_lps_to_ras_nii(nii_bytes)

                # with open("test_data/original_lps.nii.gz", "rb") as f:
                #     nii_bytes = f.read()
                logging.info(f"✅ NIfTI 변환 완료 ({len(nii_bytes) / 1024:.1f} KB)")
            except Exception as e:
                logging.exception("❌ NIfTI 변환 실패")
                return jsonify({"success": False, "message": f"NIfTI 변환 실패: {str(e)}"}), 500

            # 4. SMC 추론 요청
            encoder = MultipartEncoder(
                fields={"file": ("converted.nii.gz", BytesIO(nii_bytes), "application/octet-stream")}
            )
            monitor = MultipartEncoderMonitor(encoder, lambda m: None)
            headers = {"Content-Type": monitor.content_type}

            try:
                logging.info("📡 SMC 서버로 추론 요청 시작")
                smc_res = requests.post(
                    "https://smc-ssiso-ai.ngrok.app/infer/hcc-pv/?output_format=.nrrd",
                    data=monitor,
                    headers=headers,
                    timeout=(30, 300),
                )
                elapsed = round(time.time() - start_time, 2)

                if smc_res.status_code != 200:
                    logging.error(f"❌ SMC 서버 오류: {smc_res.status_code}, {smc_res.text}")
                    return jsonify({"success": False, "message": f"SMC 오류: {smc_res.text}", "elapsed": elapsed}), 500

                logging.info(f"✅ SMC 응답 완료 ({len(smc_res.content) / 1024:.1f} KB), 처리시간: {elapsed}s")
                # converted_nrrd_rps = convert_nrrd_to_rps(smc_res.content)

                # 5. zip 묶기

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr("converted.nii.gz", nii_bytes)
                    zip_file.writestr("inferred.nrrd", smc_res.content)  # 변경된 파일명
                    # zip_file.writestr("inferred.nrrd", converted_nrrd_rps)  # 변경된 파일명
                zip_buffer.seek(0)

                logging.info("📤 zip 파일 생성 및 클라이언트 응답")

                return send_file(
                    zip_buffer,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name="result_bundle.zip",
                )

            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                logging.exception("❌ SMC 요청 실패 예외 발생")
                return jsonify({"success": False, "message": f"SMC 요청 실패: {str(e)}", "elapsed": elapsed}), 500


def convert_to_nifti(dicom_dir: str, output_path: str):
    slices = []
    for filename in sorted(os.listdir(dicom_dir)):
        if filename.lower().endswith(".dcm"):
            path = os.path.join(dicom_dir, filename)
            try:
                ds = pydicom.dcmread(path)
                slices.append(ds)
            except Exception as e:
                logging.warning(f"❌ DICOM 읽기 실패: {filename} - {e}")
                continue

    if not slices:
        raise ValueError("DICOM 파일이 없습니다.")

    # z축 정렬
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 3D 배열 구성 (Z, Y, X)
    volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    logging.info(f"🧊 3D 볼륨 shape (before transpose): {volume.shape}")
    volume = np.transpose(volume, (2, 1, 0))  # (Z, Y, X) → (X, Y, Z)
    logging.info(f"↔️ 3D 볼륨 shape (after transpose): {volume.shape}")

    # spacing
    try:
        pixel_spacing = [float(x) for x in slices[0].PixelSpacing]
        z_spacing = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
        spacing = (pixel_spacing[0], pixel_spacing[1], z_spacing)
    except Exception as e:
        logging.warning(f"⚠ spacing 오류: {e}")
        spacing = (1.0, 1.0, 1.0)
    logging.info(f"📏 spacing (X, Y, Z): {spacing}")

    # 방향 행렬 및 원점 (affine)
    try:
        iop = slices[0].ImageOrientationPatient
        logging.info(f"🧭 ImageOrientationPatient: {iop}")
        row = np.array(iop[:3])
        col = np.array(iop[3:])
        normal = np.cross(row, col)

        direction = np.array([
            row * spacing[0],
            col * spacing[1],
            normal * spacing[2]
        ]).T
        logging.info(f"🧮 방향 행렬 (LPS):\n{direction}")

        origin = np.array(slices[0].ImagePositionPatient)
        logging.info(f"📍 ImagePositionPatient (LPS): {origin}")

        # 👉 LPS → RAS 변환
        direction, origin = convert_lps_to_ras(direction, origin)
        logging.info(f"🧮 방향 행렬 (RAS):\n{direction}")
        logging.info(f"📍 Origin (RAS): {origin}")

        affine = np.eye(4)
        affine[:3, :3] = direction
        affine[:3, 3] = origin
        logging.info(f"📐 affine:\n{affine}")
    except Exception as e:
        logging.warning(f"⚠ Affine 계산 실패: {e}")
        affine = np.diag([*spacing, 1.0])

    # NIfTI 생성
    nifti_img = nib.Nifti1Image(volume, affine)

    # 헤더 세팅
    hdr = nifti_img.header
    hdr['pixdim'][0] = 1.0  # qfac
    hdr['pixdim'][1:4] = list(spacing)

    # quaternion 기본 회전 (단위)
    nifti_img.set_qform(affine, code=1)
    hdr['qform_code'] = 1
    hdr['quatern_b'] = 0.0
    hdr['quatern_c'] = 0.0
    hdr['quatern_d'] = 1.0
    hdr['qoffset_x'] = origin[0]
    hdr['qoffset_y'] = origin[1]
    hdr['qoffset_z'] = origin[2]

    # sform 설정
    nifti_img.set_sform(affine, code=1)
    hdr['sform_code'] = 1
    hdr['srow_x'] = affine[0].astype(np.float32)
    hdr['srow_y'] = affine[1].astype(np.float32)
    hdr['srow_z'] = affine[2].astype(np.float32)

    hdr['xyzt_units'] = 10  # millimeters

    # 저장
    nib.save(nifti_img, output_path)
    logging.info(f"💾 NIfTI 저장 완료: {output_path}")

    # 헤더 로깅용 정제
    hdr_dict = {k: str(hdr[k]) for k in hdr.keys()}
    logging.info(f"🧾 NIfTI 헤더:\n{json.dumps(hdr_dict, indent=2)}")



def convert_lps_to_ras_nii(nii_bytes):
    """gzip 압축된 NIfTI(.nii.gz)를 LPS → RAS로 변환하여 gzip bytes 반환"""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp_in:
        tmp_in.write(nii_bytes)
        tmp_in.flush()

        # 1. nibabel 로딩
        img = nib.load(tmp_in.name)
        data = img.get_fdata()
        affine = img.affine.copy()

        logging.info("📌 원본 데이터 shape: %s", data.shape)
        logging.info("📌 변환 전 affine:\n%s", np.array_str(affine, precision=4, suppress_small=True))

        # affine[0, 0] = abs(affine[0, 0])   # R (+X)
        # affine[0, 3] = -abs(affine[0, 3])

        # affine[1, 1] = abs(affine[1, 1])   # A (+Y)
        # affine[1, 3] = -abs(affine[1, 3])

        # affine[2, 2] = abs(affine[2, 2])   # S (+Z)
        # affine[2, 3] = -abs(affine[2, 3])
        logging.info("🔁 Y축 반전 적용 후 affine:\n%s", np.array_str(affine, precision=4, suppress_small=True))

        # 4. NIfTI 객체 생성
        new_img = nib.Nifti1Image(data.astype(np.int16), affine)  # 원하는 bitpix 맞추기 위해 int16으로 변경
        new_img.set_qform(affine, code=1)
        new_img.set_sform(affine, code=1)
        logging.info("🧱 qform/sform 적용 완료")

        # 5. quaternion 초기화 (회전 제거)
        new_img.header['quatern_b'] = 0.0
        new_img.header['quatern_c'] = 0.0
        new_img.header['quatern_d'] = 1.0
        logging.info("🧭 쿼터니언 회전 제거 완료")

        # 6. 픽셀 크기 방향 양수 보정
        new_img.header['pixdim'][1:4] = np.abs(new_img.header['pixdim'][1:4])
        logging.info("📐 pixdim 보정: %s", new_img.header['pixdim'][1:4])

        # 7. 단위 설정
        new_img.header['xyzt_units'] = 10  # mm 단위
        logging.info("📏 단위 설정 완료 (mm)")

        # 8. 최종 header 로그
        log_nii_header(new_img)

        # 9. gzip 압축하여 반환
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp_out:
            nib.save(new_img, tmp_out.name)
            tmp_out.seek(0)
            return tmp_out.read()

def log_nii_header(img):
    hdr = img.header

    header_dict = {
        "sizeof_hdr": int(hdr["sizeof_hdr"]),
        "datatype": str(int(hdr["datatype"])),
        "bitpix": str(int(hdr["bitpix"])),
        "dim": np.array2string(hdr["dim"]),
        "pixdim": np.array2string(hdr["pixdim"]),
        "qform_code": str(int(hdr["qform_code"])),
        "sform_code": str(int(hdr["sform_code"])),
        "qoffset_x": str(hdr["qoffset_x"]),
        "qoffset_y": str(hdr["qoffset_y"]),
        "qoffset_z": str(hdr["qoffset_z"]),
        "quatern_b": str(hdr["quatern_b"]),
        "quatern_c": str(hdr["quatern_c"]),
        "quatern_d": str(hdr["quatern_d"]),
        "srow_x": np.array2string(hdr["srow_x"]),
        "srow_y": np.array2string(hdr["srow_y"]),
        "srow_z": np.array2string(hdr["srow_z"]),
        "magic": str(hdr["magic"]),
        "xyzt_units": str(int(hdr["xyzt_units"])),
        "vox_offset": str(hdr["vox_offset"]),
        "scl_slope": str(hdr["scl_slope"]),
        "scl_inter": str(hdr["scl_inter"]),
        "intent_code": str(int(hdr["intent_code"])),
        "intent_name": str(hdr["intent_name"]),
        "intent_p1": str(hdr["intent_p1"]),
        "intent_p2": str(hdr["intent_p2"]),
        "intent_p3": str(hdr["intent_p3"]),
        "descrip": str(hdr["descrip"]),
        "aux_file": str(hdr["aux_file"]),
        "glmax": str(int(hdr["glmax"])),
        "glmin": str(int(hdr["glmin"])),
    }

    logging.info("✅ 변환 후 NIfTI header:\n%s", json.dumps(header_dict, indent=2))


def convert_nrrd_to_rps(nrrd_bytes):
    """NRRD 파일의 X, Y축 (LPS→RAS) 방향 반전하여 bytes 반환"""
    logger = logging.getLogger(__name__)
    logger.info("📥 NRRD 바이트 입력 수신, 크기: %.2f KB", len(nrrd_bytes) / 1024)

    # 임시 파일로 저장해서 읽기
    with tempfile.NamedTemporaryFile(suffix=".nrrd") as temp_in:
        temp_in.write(nrrd_bytes)
        temp_in.flush()
        logger.info("📄 임시 NRRD 파일 저장 위치: %s", temp_in.name)

        data, header = nrrd.read(temp_in.name)
        logger.info("📊 NRRD shape: %s, dtype: %s", data.shape, data.dtype)
        logger.info("🧭 원본 방향 정보: %s", header.get('space directions'))
        logger.info("🧭 원본 origin 정보: %s", header.get('space origin'))

    # ✅ X축 + Y축 반전 (좌우 + 앞뒤)
    # data = data[:, ::-1, :]
    # data = data[::-1, :, :]
    # data = data[::-1, ::-1, :]
    # data = data[:, :, ::-1]
    # data = data[::-1, :, ::-1]
    # ✅ 방향 정보 수정
    # if 'space directions' in header and isinstance(header['space directions'][0], tuple):
    #     direction = list(header['space directions'])
    #     direction[0] = tuple([-v for v in direction[0]])  # X
    #     # direction[1] = tuple([-v for v in direction[1]])  # Y
    #     direction[2] = tuple([-v for v in direction[2]])  # Z
    #     header['space directions'] = tuple(direction)
    #     logger.info("🧭 수정된 방향 정보: %s", header['space directions'])

    # # ✅ origin 수정
    # if 'space origin' in header:
    #     origin = list(header['space origin'])
    #     origin[0] = -origin[0]
    #     # origin[1] = -origin[1]
    #     origin[2] = -origin[2]
    #     header['space origin'] = tuple(origin)
    #     logger.info("🧭 수정된 origin 정보: %s", header['space origin'])

    # 결과를 gzip 압축된 NRRD로 다시 저장
    out_io = BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".nrrd") as temp_out:
        nrrd.write(temp_out.name, data, header)
        temp_out.seek(0)
        raw_output = temp_out.read()
        out_io.write(raw_output)
        logger.info("📤 변환 완료 NRRD 크기: %.2f KB", len(raw_output) / 1024)

    return out_io.getvalue()





@app.route("/inspect-nifti", methods=["POST"])
def inspect_nifti():
    try:
        file = request.files["file"]

        # 파일을 임시로 저장 후 load
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmpfile:
            tmpfile.write(file.read())
            tmpfile.flush()

            nifti_img = nib.load(tmpfile.name)  # 경로 기반으로 로드
            header = nifti_img.header
            header_info = {key: str(header[key]) for key in header.keys()}

        return jsonify({"success": True, "header": header_info})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400










def convert_lps_to_ras(direction: np.ndarray, origin: np.ndarray):
    """
    DICOM LPS 좌표계를 RAS 좌표계로 변환.
    방향 행렬과 origin의 X, Y축 부호를 반전.
    """
    direction_ras = direction.copy()
    origin_ras = origin.copy()
    
    # X (L→R), Y (P→A) 방향 반전
    direction_ras[:, 0] *= -1
    direction_ras[:, 1] *= -1
    origin_ras[0] *= -1
    origin_ras[1] *= -1

    return direction_ras, origin_ras









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
            print("stdout is TTY:", sys.stdout.isatty())
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
    progress = tqdm(
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc="Uploading to SMC",
        ncols=80,
        dynamic_ncols=True,
        ascii=True,         # ascii 바 (#)로 표시 – 호환성 보장
        file=sys.stdout,    # 강제로 stdout 지정
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
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
        print(f"\n가장 추정되는 위상: {main_phase}")
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

def make_obj_and_mtl_for_4d(data, segment_infos, output_dir, folder_name):
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
        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.001, iterations=10)
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
        print(f"기존 파일을 사용합니다. ({len(obj_urls)}개)")
        return jsonify({
            "success": True,
            "objUrls": obj_urls,
            "mtlUrl": f"/uploads/meshes/{folder_name}/segments.mtl"
        })

    # 2. 새로 저장 및 변환
    uploaded_file.save(input_path)
    print(f"📄 저장된 NRRD: {input_path}")
    data, header = nrrd.read(input_path)
    print("Spacing (space directions):", header.get("space directions"))

    segment_infos = parse_slicer_segment_infos(header)
    print("=== Slicer 세그먼트 정보 ===")
    for idx, info in sorted(segment_infos.items()):
        print(f"  Segment{idx}: label={info.get('label')}, name={info.get('name')}, color={info.get('color')}")

    obj_urls, mtl_lines = [], []
    if data.ndim == 4:
        print(f"[INFO] Detected 4D NRRD → 3D label map 변환 중...")
        data3d = convert_4d_nrrd_to_3d_labelmap(data, segment_infos)
        obj_urls, mtl_lines = make_obj_and_mtl(data3d, header, segment_infos, target_dir, folder_name)
    elif data.ndim == 3:
        obj_urls, mtl_lines = make_obj_and_mtl_for_3d(data, segment_infos, output_dir, folder_name)
    else:
        print("❗지원하지 않는 NRRD shape:", data.shape)

    # MTL 저장
    with open(mtl_path, "w") as f:
        f.write("\n".join(mtl_lines))

    print(f"총 {len(obj_urls)}개 세그먼트 색상 적용 완료")
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

def convert_4d_nrrd_to_3d_labelmap(data4d, segment_infos):
    """
    4D NRRD (C, X, Y, Z)를 3D label map (X, Y, Z)으로 변환
    segment_infos: Segment1_LabelValue 등 정보 활용 (없으면 index+1 사용)
    """
    print(f"[convert_4d_nrrd_to_3d_labelmap] input shape: {data4d.shape}")

    num_channels = data4d.shape[0]
    out_shape = data4d.shape[1:]
    label_map = np.zeros(out_shape, dtype=np.uint8)

    # 각 채널별 label 값 추출
    channel_label_values = []
    for ch in range(num_channels):
        info = segment_infos.get(ch)
        if info and 'label' in info:
            label_val = int(info['label'])
        else:
            label_val = ch + 1
        channel_label_values.append(label_val)

    # 우선순위: 채널 순서대로 → 나중 채널이 덮어씀 (원하면 반대 처리도 가능)
    for ch in range(num_channels):
        ch_mask = data4d[ch] > 0
        label_val = channel_label_values[ch]
        label_map[ch_mask] = label_val

    print(f"[convert_4d_nrrd_to_3d_labelmap] output unique labels: {np.unique(label_map)}")
    return label_map


def send_to_smc(nii_path):
    file_size = os.path.getsize(nii_path)
    progress = tqdm(
        total=file_size,
        unit='B',
        unit_scale=True,
        desc="Uploading to SMC",
        ascii=True,
        ncols=80,
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    def callback(monitor):
        progress.update(monitor.bytes_read - callback.last_bytes)
        callback.last_bytes = monitor.bytes_read
    callback.last_bytes = 0

    encoder = MultipartEncoder(
        fields={"file": ("converted.nii.gz", open(nii_path, "rb"), "application/octet-stream")}
    )
    monitor = MultipartEncoderMonitor(encoder, callback)

    headers = {"Content-Type": monitor.content_type}

    response = requests.post(
        "https://smc-ssiso-ai.ngrok.app/infer/hcc-pv/?output_format=.nrrd",
        data=monitor,
        headers=headers,
        timeout=(30, 300),
    )

    progress.close()
    return response

def assign_fallback_color(index):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    rgb = cmap(index % 20)[:3]
    return [round(c, 3) for c in rgb]

import re
def parse_slicer_segment_infos(header):
    result = {}
    for key, val in header.items():
        if not isinstance(key, str):
            continue
        match = re.match(r"Segment(\d+)_", key)
        if not match:
            continue
        seg_id = int(match.group(1))
        subkey = key.split("_", 1)[1]
        if seg_id not in result:
            result[seg_id] = {}
        result[seg_id][subkey.lower()] = val

    for seg_id, info in result.items():
        color = info.get("color")
        if isinstance(color, str):
            try:
                color = list(map(float, color.strip().split()))
            except:
                color = None
        elif isinstance(color, list):
            color = [float(c) for c in color]
        else:
            color = None

        if not color or len(color) < 3:
            color = assign_fallback_color(seg_id)
            print(f"Segment{seg_id} → fallback color: {color}")

        # ⭐ [핵심] 항상 0~1 스케일로 맞추기
        if max(color) > 1.0:
            color = [c / 255 for c in color]

        info["color"] = color

        if "label" not in info or info["label"] is None:
            info["label"] = seg_id + 1

    if not result:
        print("❗ 헤더 세그먼트 없음 → fallback 생성")
        for i in range(1, 31):
            result[i] = {
                "label": i,
                "name": f"Segment{i}",
                "color": assign_fallback_color(i)
            }

    return result

def assign_fallback_color(index):
    """기본 색상 팔레트에서 색상 선택"""
    cmap = plt.get_cmap('tab20')  # 최대 20개까지 확실하게 구분됨
    color = cmap(index % 20)[:3]  # RGBA 중 RGB만 사용
    return [round(c, 3) for c in color]  # 0~1 범위로 반올림

def make_obj_and_mtl(data, header, segment_infos, out_dir, folder_name):
    obj_urls = []
    mtl_lines = []
    labels = np.unique(data)

    mesh_output_dir = out_dir
    directions = header.get("space directions", None)
    if directions is not None:
        spacing = []
        flips = []
        for vec in directions:
            vec = np.array(vec)
            norm = np.linalg.norm(vec)
            spacing.append(norm)

            # 가장 큰 축 방향의 sign으로 flip 여부 결정
            max_axis = np.argmax(np.abs(vec))
            flip_sign = np.sign(vec[max_axis])
            flips.append(flip_sign)

        # print(f">>> spacing used for mesh: {spacing}")
        # print(f">>> flips used for mesh: {flips}")
    else:
        spacing = [1.0, 1.0, 1.0]
    # print(">>> spacing used for mesh:", spacing)

    # === HCC-PV label → name 매핑 ===
    label_name_mapping = {
        1: "Liver",
        2: "Rt.lobe",
        3: "RAS",
        4: "RPS",
        5: "Lt.lobe",
        6: "LLS",
        7: "LMS",
        8: "Spigelian",
        9: "PV",
        10: "HV",
        11: "Cancer",
        12: "BD"
    }

    # === HCC-PV label → color 매핑 === (RGB 0~1)
    label_color_mapping = {
        1: [238/255, 112/255, 70/255],
        2: [238/255, 112/255, 70/255],
        3: [218/255, 108/255, 110/255],
        4: [138/255, 117/255, 231/255],
        5: [211/255, 255/255, 51/255],
        6: [255/255, 147/255, 77/255],
        7: [185/255, 202/255, 99/255],
        8: [79/255, 255/255, 174/255],
        9: [193/255, 157/255, 255/255],
        10: [139/255, 186/255, 255/255],
        11: [234/255, 36/255, 36/255],
        12: [95/255, 170/255, 127/255]
    }

    for label in labels:
        if label == 0:
            continue  # background는 제외

        label_int = int(label)
        # === 이름 / 색상 가져오기
        name = label_name_mapping.get(label_int, f"Segment{label_int}")
        color = label_color_mapping.get(label_int, [0.5, 0.5, 0.5])

        # === 출력용 확인
        # print(f"[Label={label_int}] name={name}, color={color}")

        # material 이름 및 파일명 통일
        material_name = f"segment_{name.replace(' ', '_')}"
        obj_filename = f"{material_name}.obj"
        obj_path = os.path.join(mesh_output_dir, obj_filename)

        # 마스크 생성
        mask = (data == label)
        mask_sum = np.sum(mask)
        # print(f"\n[Label={label_int}] mask sum = {mask_sum}")

        # if mask_sum < 1000:
        #     print(f"[Label={label_int}] mask too small → skip")
        #     continue

        # print(f"[Label={label_int}] marching_cubes 시작...")
        origin = header.get("space origin", [0.0, 0.0, 0.0])
        # 메쉬 저장 (material_name 포함)
        save_mesh_from_volume(mask, obj_path, material_name, spacing=spacing, origin=origin, flips=flips)
        # print(f"Label {label_int}: mask sum = {np.sum(mask)}")
        # MTL 정의 추가
        mtl_lines.append(f"newmtl {material_name}")
        mtl_lines.append(f"Kd {color[0]} {color[1]} {color[2]}")
        mtl_lines.append("Ka 0 0 0")
        mtl_lines.append("Ks 0 0 0")
        mtl_lines.append("d 1.0")
        mtl_lines.append("illum 1")
        mtl_lines.append("")

        # URL 등록
        obj_urls.append(f"/uploads/dicom/{folder_name}/{obj_filename}")

    return obj_urls, mtl_lines

def save_mesh_from_volume(
    binary_volume: np.ndarray,
    obj_path: str,
    material_name: str = "default",
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    flips=(1.0, 1.0, 1.0),
    smoothing_iterations: int = 10
):
    if not np.any(binary_volume):
        print(f"skip: 비어있는 볼륨입니다 -> {obj_path}")
        return

    verts, faces, normals, _ = measure.marching_cubes(binary_volume, level=0.5, spacing=spacing)

    # print(f"[save_mesh_from_volume] verts.shape = {verts.shape}")
    # print(f"[save_mesh_from_volume] faces.shape = {faces.shape}")

    # origin 적용만 사용 (flips는 일반적으로 사용 안 함 → 방향이 꼬임)
    verts = verts + np.array(origin)
    # print(f"[save_mesh_from_volume] affine 적용 후 verts min = {verts.min(axis=0)}, max = {verts.max(axis=0)}")

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    # if np.sum(binary_volume) > 3000:
    #     print(f"[save_mesh_from_volume] smoothing 적용 시작")
    #     mesh = trimesh.smoothing.filter_laplacian(    
    #         mesh,
    #         lamb=0.1,
    #         iterations=10,
    #         implicit_time_integration=True,
    #         volume_constraint=True
    #     )
    #     print(f"[save_mesh_from_volume] smoothing 적용 완료")
    # else:
    #     print(f"[save_mesh_from_volume] smoothing 적용 안 함 (작은 segment)")
    # print(f"[save_mesh_from_volume] smoothing 적용 시작")
    mesh = trimesh.smoothing.filter_laplacian(
        mesh,
        lamb=0.1,
        iterations=5,
        implicit_time_integration=True,
        volume_constraint=True
    )
    # print(f"[save_mesh_from_volume] smoothing 적용 완료")
    mesh.export(obj_path)
    # print(f"[save_mesh_from_volume] OBJ 저장 완료: {obj_path}")

    # MTL 적용
    with open(obj_path, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"mtllib segments.mtl\nusemtl {material_name}\n" + content)

@app.route("/upload-and-infer-all", methods=["POST"])
def upload_and_infer_all():
    t_start = time.time()

    folder_name = request.form.get("folder")
    if not folder_name:
        return jsonify({"success": False, "message": "folder 이름이 필요합니다"}), 400

    target_dir = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"[1/6] 폴더 생성 확인: {target_dir}")
    # DICOM 업로드 (이미 존재하는 경우 건너뜀)
    files = request.files.getlist("dicomFiles")
    saved = 0
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(target_dir, filename)
        if not os.path.exists(save_path):
            file.save(save_path)
            saved += 1
    print(f"[2/6] DICOM {saved}/{len(files)}개 저장 완료")
    # NIfTI 변환
    nii_path = os.path.join(target_dir, "converted.nii.gz")
    if not os.path.exists(nii_path):
        t1 = time.time()
        convert_to_nifti(target_dir, nii_path)
        print(f"[3/6] NIfTI 변환 완료 ({round(time.time() - t1, 2)}초)")
    else:
        print(f"[3/6] NIfTI 이미 존재: {nii_path}")

    # 추론 (SMC 호출)
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

    # ★ segment_infos 항상 미리 로드하기 → NameError 방지
    data, header = nrrd.read(nrrd_path)
    segment_infos = parse_slicer_segment_infos(header)

    labels = np.unique(data)
    label_name_mapping = {
        1: "Liver",
        2: "Rt.lobe",
        3: "RAS",
        4: "RPS",
        5: "Lt.lobe",
        6: "LLS",
        7: "LMS",
        8: "Spigelian",
        9: "PV",
        10: "HV",
        11: "Cancer",
        12: "BD"
    }

    print("\n[INFO] 포함된 세그먼트 목록:")
    for label in labels:
        if label == 0:
            continue  # background 생략
        name = label_name_mapping.get(label, f"UnknownLabel{label}")
        print(f"  - Label {int(label)}: {name}")

    # NRRD → OBJ/MTL 변환
    obj_urls = []
    mtl_lines = []
    mtl_path = os.path.join(target_dir, "segments.mtl")

    if os.path.exists(mtl_path):
        print("[5/6] OBJ/MTL 파일이 이미 존재합니다. 건너뜀")
        for file in os.listdir(target_dir):
            if file.endswith(".obj"):
                obj_urls.append(f"/uploads/dicom/{folder_name}/{file}")
        with open(mtl_path, "r") as f:
            mtl_lines = f.read().splitlines()
    else:
        t3 = time.time()
        obj_urls, mtl_lines = make_obj_and_mtl(data, header, segment_infos, target_dir, folder_name)
        print(f"[5/6] OBJ 세그먼트 {len(obj_urls)}개 생성 완료 ({round(time.time() - t3, 2)}초)")

        with open(mtl_path, "w") as f:
            f.write("\n".join(mtl_lines))
        print(f"[6/6] MTL 저장 완료: {mtl_path}")

    total_sec = round(time.time() - t_start, 2)
    print(f"[완료] 전체 처리 시간: {total_sec}초")

    # labelColorMap 구성
    labelColorMap = {}
    for idx, info in segment_infos.items():
        label_name = info.get("name", f"Segment{idx}")
        color = info.get("color", [0.5, 0.5, 0.5])
        color255 = [int(c * 255) for c in color] + [255]  # alpha 255 고정
        labelColorMap[f"{label_name}"] = color255

    return jsonify({
        "success": True,
        "niiUrl": f"/uploads/dicom/{folder_name}/converted.nii.gz",
        "nrrdUrl": f"/uploads/dicom/{folder_name}/inferred.nrrd",
        "objUrls": obj_urls,
        "mtlUrl": f"/uploads/dicom/{folder_name}/segments.mtl",
        "labelColorMap": labelColorMap,
        "volumeTable": {
            "columns": [],
            "rows": []
    }
    })

app.run(host="0.0.0.0", port=5051, debug=False, use_reloader=True)