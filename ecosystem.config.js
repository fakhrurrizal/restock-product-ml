module.exports = {
  apps: [
    {
      name: "python-api",
      script: "venv/bin/python",
      args: "main.py",
      cwd: "./backend",
      interpreter: "",
    },
    {
      name: "next-frontend",
      script: "yarn",
      args: "start",
      cwd: "./frontend",
      env: {
        NODE_ENV: "production",
        PORT: 3000
      }
    }
  ]
};