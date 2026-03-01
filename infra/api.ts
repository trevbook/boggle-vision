// ── Models bucket ────────────────────────────────────────────────────────────
// SST's Python container builder only copies .py files into the Docker build
// context, so model weights can't be baked into the image. Instead we store
// them in S3 and download to /tmp on cold start.
const modelsBucket = new sst.aws.Bucket("ModelsBucket");

const modelFiles = [
  "prototyping/yolov8s-seg.pt",
  "prototyping/legacy/models/boggle_cnn_v2.onnx",
  "prototyping/legacy/models/boggle_cnn_v2.onnx.data",
];

for (const filePath of modelFiles) {
  const key = filePath.split("/").pop() as string;
  new aws.s3.BucketObjectv2(`Model-${key}`, {
    bucket: modelsBucket.name,
    key,
    source: new $util.asset.FileAsset(filePath),
  });
}

// ── API ──────────────────────────────────────────────────────────────────────
export const api = new sst.aws.ApiGatewayV2("Api", {
  cors: {
    allowMethods: ["POST", "OPTIONS"],
    allowHeaders: ["Content-Type"],
    allowOrigins: ["*"],
  },
});

api.route("POST /analyze", {
  handler: "./cv_pipeline/handler.handler",
  runtime: "python3.12",
  python: {
    container: true,
  },
  link: [modelsBucket],
  environment: {
    // S3 bucket for deployed Lambda (models downloaded to /tmp on cold start)
    MODELS_BUCKET: modelsBucket.name,
    // Absolute path for sst dev (local files, S3 not used)
    MODELS_DIR: `${process.cwd()}/cv_pipeline/models`,
  },
  memory: "2048 MB",
  timeout: "60 seconds",
  architecture: "x86_64",
});
