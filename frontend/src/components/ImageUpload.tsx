import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface ImageUploadProps {
  onFileSelect: (file: File, previewUrl: string) => void;
  disabled?: boolean;
}

export default function ImageUpload({ onFileSelect, disabled }: ImageUploadProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];
      if (file) {
        onFileSelect(file, URL.createObjectURL(file));
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept: { "image/*": [".png", ".jpg", ".jpeg", ".bmp", ".tiff"] },
      maxFiles: 1,
      disabled,
    });

  return (
    <div className="space-y-2">
      <div
        {...getRootProps()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition
          ${isDragActive ? "border-primary-500 bg-primary-50" : "border-slate-300 hover:border-primary-400 hover:bg-slate-50"}
          ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        <input {...getInputProps()} />
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-primary-100">
          <svg className="h-7 w-7 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <p className="text-sm font-medium text-slate-700">
          {isDragActive ? "Drop X-ray here" : "Drag & drop pediatric hand X-ray"}
        </p>
        <p className="mt-1 text-xs text-slate-500">PNG, JPEG, BMP — max 20 MB</p>
      </div>
      {fileRejections.length > 0 && (
        <p className="text-sm text-red-600">
          {fileRejections[0].errors[0]?.message ?? "Invalid file"}
        </p>
      )}
    </div>
  );
}
