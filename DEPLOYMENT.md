# Deployment Guide

This guide covers deploying NeuroDecode for portfolio/demo purposes using free hosting tiers.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Vercel        │     │    Render       │
│   (Frontend)    │────▶│   (Backend)     │
│   React/Vite    │     │   FastAPI       │
│   FREE tier     │     │   FREE tier     │
└─────────────────┘     └─────────────────┘
```

## Quick Deploy

### 1. Deploy Backend to Render

1. **Create a Render account** at [render.com](https://render.com)

2. **Connect your GitHub repository**
   - Go to Dashboard → New → Web Service
   - Connect your GitHub account and select this repository

3. **Configure the service**
   - **Name**: `neurodecode-api`
   - **Region**: Oregon (US West)
   - **Branch**: `main`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Build Context**: `.`

4. **Set environment variables**
   ```
   ENVIRONMENT=production
   DEBUG=false
   CORS_ORIGINS=https://your-frontend.vercel.app
   ```

5. **Deploy** - Click "Create Web Service"

6. **Note your backend URL** (e.g., `https://neurodecode-api.onrender.com`)

### 2. Deploy Frontend to Vercel

1. **Create a Vercel account** at [vercel.com](https://vercel.com)

2. **Import your repository**
   - Go to Dashboard → Add New → Project
   - Import your GitHub repository

3. **Configure the project**
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

4. **Set environment variables**
   ```
   VITE_API_URL=https://neurodecode-api.onrender.com
   VITE_WS_URL=wss://neurodecode-api.onrender.com/ws/decode
   ```

5. **Deploy** - Click "Deploy"

### 3. Update CORS (Important!)

After deploying the frontend, update the backend CORS settings:

1. Go to your Render service → Environment
2. Update `CORS_ORIGINS` to include your Vercel URL:
   ```
   CORS_ORIGINS=https://neurodecode.vercel.app,https://neurodecode-frontend.vercel.app
   ```

## Alternative: One-Click Deploy

### Render Blueprint

The `render.yaml` file enables one-click deployment:

1. Fork this repository
2. Go to [Render Dashboard](https://dashboard.render.com/blueprints)
3. Click "New Blueprint Instance"
4. Select your forked repository
5. Render will auto-detect `render.yaml` and create the service

## Environment Variables Reference

### Backend (Render)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Runtime environment | `production` |
| `DEBUG` | Enable debug mode | `false` |
| `CORS_ORIGINS` | Allowed frontend origins (comma-separated) | localhost URLs |
| `PORT` | Server port (auto-set by Render) | `8000` |

### Frontend (Vercel)

| Variable | Description | Required |
|----------|-------------|----------|
| `VITE_API_URL` | Backend API URL | Yes (production) |
| `VITE_WS_URL` | WebSocket URL | Yes (production) |

## Troubleshooting

### WebSocket Connection Fails

- Ensure `VITE_WS_URL` uses `wss://` (secure WebSocket) in production
- Check that CORS origins are correctly set on the backend
- Verify Render service is running (free tier sleeps after 15 min of inactivity)

### Backend Service Sleeping

Render's free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds to wake up.

**Solutions:**
- Use a service like [UptimeRobot](https://uptimerobot.com) to ping `/health` every 10 minutes
- Upgrade to Render's paid tier ($7/month) for always-on service

### Build Failures

**Backend:**
- Check Docker build logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`

**Frontend:**
- Check Vercel build logs
- Ensure `npm run build` works locally

## Local Development

```bash
# Backend
cd neural_decoding_sys
pip install -r requirements.txt
uvicorn src.backend.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Cost Summary

| Service | Tier | Cost | Limitations |
|---------|------|------|-------------|
| Render (Backend) | Free | $0 | Sleeps after 15 min inactivity |
| Vercel (Frontend) | Hobby | $0 | 100GB bandwidth/month |
| **Total** | | **$0** | |

## Production Considerations

For actual production use (not just portfolio), consider:

1. **Database**: Add PostgreSQL (Render: $7/month, Neon: free tier available)
2. **Redis**: Add Redis for caching (Render: $10/month, Upstash: free tier available)
3. **Always-on**: Upgrade Render to Starter ($7/month)
4. **Custom domain**: Configure in both Vercel and Render dashboards
