# Render deployment

Deploy the same Dockerized API as a Render Web Service.

1. Push the repository to GitHub.
2. In Render, create `New +` → `Web Service`.
3. Connect the `Rent-tracker` repository.
4. Choose `Docker` as the runtime.
5. Keep the repo root as the service root.
6. Set the health check path to `/health`.
7. Create the service and wait for the first deploy.

After deploy:

```bash
curl -s https://<your-render-service>.onrender.com/health
```

Notes:
- The current container exposes `/health`, `/predict`, and `/docs`.
- `models/baseline_latest/model.joblib` must be present in the built image, otherwise `/health` will be `degraded` and `/predict` will return `503`.
- If Render does not detect the port correctly, the minimal follow-up change is to make the Docker command read the `PORT` environment variable.
