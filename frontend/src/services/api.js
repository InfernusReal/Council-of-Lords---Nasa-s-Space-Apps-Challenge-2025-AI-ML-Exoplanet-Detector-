// API service for connecting to the Council of Lords backend
class CouncilAPI {
  constructor() {
    this.baseURL = import.meta.env.PROD ? 'http://localhost:8000' : '/api';
  }

  async analyzeFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseURL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Council API Error:', error);
      throw error;
    }
  }

  async getHealth() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'ERROR', brutal_reality_mode: 'UNKNOWN' };
    }
  }

  async getStatus() {
    try {
      const response = await fetch(`${this.baseURL}/`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Status check failed:', error);
      return { message: 'Council unavailable', version: 'unknown' };
    }
  }
}

export const councilAPI = new CouncilAPI();