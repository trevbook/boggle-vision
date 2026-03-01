/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "boggle-vision",
      removal: input?.stage === "production" ? "retain" : "remove",
      protect: ["production"].includes(input?.stage),
      home: "aws",
      providers: {
        aws: {
          ...(!process.env.CI && { profile: "personal" }),
        },
      },
    };
  },
  async run() {
    await import("./infra/secrets");
    await import("./infra/api");
    await import("./infra/frontend");
  },
});
