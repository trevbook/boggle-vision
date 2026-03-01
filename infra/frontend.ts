import { api } from "./api.js";
import { secrets } from "./secrets.js";

export const frontend = new sst.aws.Nextjs("www", {
  path: "apps/www",
  environment: {
    ...secrets,
    NEXT_PUBLIC_API_URL: api.url,
  },
});
