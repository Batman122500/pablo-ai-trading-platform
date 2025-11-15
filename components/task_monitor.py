import streamlit as st
import time
import threading


class TaskMonitor:
    """Monitor background tasks and their progress"""

    def __init__(self):
        self.tasks = {}

    def start_task(self, task_id, task_name):
        """Start tracking a new task"""
        self.tasks[task_id] = {
            'name': task_name,
            'start_time': time.time(),
            'progress': 0,
            'status': 'running',
            'message': ''
        }

    def update_task(self, task_id, progress, message=""):
        """Update task progress"""
        if task_id in self.tasks:
            self.tasks[task_id]['progress'] = progress
            self.tasks[task_id]['message'] = message

    def complete_task(self, task_id, message="Completed"):
        """Mark task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['progress'] = 100
            self.tasks[task_id]['message'] = message

    def fail_task(self, task_id, error_message):
        """Mark task as failed"""
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = 'failed'
            self.tasks[task_id]['message'] = error_message

    def get_task_status(self, task_id):
        """Get current task status"""
        return self.tasks.get(task_id, {})

    def render_monitor(self):
        """Render the task monitoring UI"""
        if not self.tasks:
            return

        st.markdown("---")
        st.markdown("### 🔄 Background Tasks")

        active_tasks = {k: v for k, v in self.tasks.items() if v['status'] == 'running'}

        if active_tasks:
            st.info("🟡 Background tasks are running...")

        for task_id, task_info in self.tasks.items():
            with st.expander(f"{task_info['name']} - {task_info['status'].title()}",
                             expanded=task_info['status'] == 'running'):
                col1, col2 = st.columns([3, 1])

                with col1:
                    status_icon = "🟢" if task_info['status'] == 'completed' else "🔴" if task_info[
                                                                                            'status'] == 'failed' else "🟡"
                    st.write(f"{status_icon} **{task_info['name']}**")
                    if task_info['message']:
                        st.caption(task_info['message'])

                with col2:
                    if task_info['status'] == 'running':
                        st.progress(task_info['progress'] / 100)
                        st.caption(f"{task_info['progress']:.0f}%")
                    else:
                        status_color = "green" if task_info['status'] == 'completed' else "red"
                        st.markdown(
                            f"<span style='color: {status_color}; font-weight: bold'>{task_info['status'].upper()}</span>",
                            unsafe_allow_html=True)

                if task_info['status'] in ['completed', 'failed']:
                    if st.button("Clear", key=f"clear_{task_id}"):
                        del self.tasks[task_id]
                        st.rerun()


# Global task monitor instance
task_monitor = TaskMonitor()


def show_task_monitor():
    """Display the task monitor component"""
    task_monitor.render_monitor()