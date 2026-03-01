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
  environment: {
    // Absolute path for sst dev (local files; deployed Lambda uses /opt/models baked into base image)
    MODELS_DIR: `${process.cwd()}/cv_pipeline/models`,
  },
  memory: "3008 MB",
  timeout: "120 seconds",
  architecture: "arm64",
  dev: false,
});
