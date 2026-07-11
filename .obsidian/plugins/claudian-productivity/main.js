const { Plugin, Modal, Setting, Notice, TFile } = require('obsidian');

const TASK_INBOX = '00 Home/Tasks/Task Inbox.md';
const TASK_CENTER = '00 Home/Tasks/Task Command Center.md';
const TASK_KANBAN = '00 Home/Tasks/Task Kanban Board.md';
const DAILY_FOLDER = '00 Home/Daily Notes';

function today() {
  return window.moment ? window.moment().format('YYYY-MM-DD') : new Date().toISOString().slice(0, 10);
}

function clean(s) { return String(s || '').trim(); }

class TaskCaptureModal extends Modal {
  constructor(app, defaults, onSubmit) {
    super(app);
    this.defaults = defaults || {};
    this.onSubmit = onSubmit;
    this.values = {
      text: '',
      tag: this.defaults.tag || '#task/inbox',
      priority: this.defaults.priority || '',
      due: '',
      scheduled: '',
      reminder: ''
    };
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl('h2', { text: this.defaults.title || 'Capture task' });

    new Setting(contentEl)
      .setName('Task')
      .setDesc('What needs to happen?')
      .addTextArea(t => {
        t.setPlaceholder('Review paper X / Finish HW problem / Email professor...')
          .onChange(v => this.values.text = v);
        t.inputEl.rows = 3;
        t.inputEl.focus();
      });

    new Setting(contentEl)
      .setName('Tag')
      .setDesc('Used by task queries and dashboards.')
      .addText(t => t.setValue(this.values.tag).onChange(v => this.values.tag = v));

    new Setting(contentEl)
      .setName('Priority')
      .setDesc('Optional: ⏫ highest, 🔺 high, 🔼 medium, 🔽 low, ⏬ lowest')
      .addText(t => t.setPlaceholder('🔺').onChange(v => this.values.priority = v));

    new Setting(contentEl)
      .setName('Due date')
      .setDesc('Optional YYYY-MM-DD; writes 📅 date')
      .addText(t => t.setPlaceholder(today()).onChange(v => this.values.due = v));

    new Setting(contentEl)
      .setName('Scheduled date')
      .setDesc('Optional YYYY-MM-DD; writes ⏳ date')
      .addText(t => t.setPlaceholder(today()).onChange(v => this.values.scheduled = v));

    new Setting(contentEl)
      .setName('Reminder')
      .setDesc('Optional YYYY-MM-DD HH:mm; writes Reminder syntax')
      .addText(t => t.setPlaceholder(`${today()} 09:00`).onChange(v => this.values.reminder = v));

    new Setting(contentEl)
      .addButton(b => b.setButtonText('Capture to Inbox').setCta().onClick(async () => {
        await this.submit();
      }))
      .addButton(b => b.setButtonText('Cancel').onClick(() => this.close()));
  }

  async submit() {
    const text = clean(this.values.text);
    if (!text) {
      new Notice('Task is empty.');
      return;
    }
    await this.onSubmit(this.values);
    this.close();
  }
}

module.exports = class ClaudianProductivityPlugin extends Plugin {
  async onload() {
    this.addCommand({ id: 'open-task-command-center', name: 'Open Task Command Center', callback: () => this.openPath(TASK_CENTER) });
    this.addCommand({ id: 'open-task-kanban-board', name: 'Open Task Kanban Board', callback: () => this.openPath(TASK_KANBAN) });
    this.addCommand({ id: 'open-task-inbox', name: 'Open Task Inbox', callback: () => this.openPath(TASK_INBOX) });
    this.addCommand({ id: 'open-today-daily-note', name: "Open today's daily note", callback: () => this.openPath(`${DAILY_FOLDER}/${today()}.md`, true) });
    this.addCommand({ id: 'capture-task-inbox', name: 'Capture task to Inbox', callback: () => this.capture({ title: 'Capture task', tag: '#task/inbox' }) });
    this.addCommand({ id: 'capture-research-task', name: 'Capture research task', callback: () => this.capture({ title: 'Capture research task', tag: '#task/research' }) });
    this.addCommand({ id: 'capture-homework-task', name: 'Capture homework task', callback: () => this.capture({ title: 'Capture homework task', tag: '#task/homework' }) });
    this.addCommand({ id: 'capture-admin-task', name: 'Capture admin task', callback: () => this.capture({ title: 'Capture admin task', tag: '#task/admin' }) });
    this.addRibbonIcon('check-square', 'Task Command Center', () => this.openPath(TASK_CENTER));
    this.addRibbonIcon('columns-3', 'Task Kanban Board', () => this.openPath(TASK_KANBAN));
    this.addRibbonIcon('list-plus', 'Capture task', () => this.capture({ title: 'Capture task', tag: '#task/inbox' }));
  }

  async ensureFile(path, content) {
    const file = this.app.vault.getAbstractFileByPath(path);
    if (file instanceof TFile) return file;
    const parent = path.split('/').slice(0, -1).join('/');
    if (parent && !this.app.vault.getAbstractFileByPath(parent)) await this.app.vault.createFolder(parent);
    return await this.app.vault.create(path, content || '');
  }

  async openPath(path, createDaily = false) {
    let file = this.app.vault.getAbstractFileByPath(path);
    if (!file && createDaily) {
      file = await this.ensureFile(path, `---\ncssclasses:\n  - dashboard\nDate: ${today()}\nStatus: In-Progress\nCategory:\n  - Daily-Note\n---\n# ${today()} Daily Note\n\n## 🎯 Top 3\n\n- [ ] \n- [ ] \n- [ ] \n\n## 📥 Captured tasks\n\n\n## 🧠 Notes\n\n`);
    }
    if (file instanceof TFile) await this.app.workspace.getLeaf(false).openFile(file);
    else new Notice(`Missing file: ${path}`);
  }

  formatTask(values) {
    const parts = ['- [ ]', clean(values.text)];
    if (clean(values.priority)) parts.push(clean(values.priority));
    if (clean(values.reminder)) parts.push(`(@${clean(values.reminder)})`);
    if (clean(values.scheduled)) parts.push(`⏳ ${clean(values.scheduled)}`);
    if (clean(values.due)) parts.push(`📅 ${clean(values.due)}`);
    if (clean(values.tag)) parts.push(clean(values.tag));
    return parts.join(' ');
  }

  async capture(defaults) {
    new TaskCaptureModal(this.app, defaults, async values => {
      const line = this.formatTask(values);
      const file = await this.ensureFile(TASK_INBOX, '# 📥 Task Inbox\n\n## Inbox\n');
      await this.app.vault.process(file, data => {
        if (!data.includes('## Inbox')) data += '\n\n## Inbox\n';
        return data.trimEnd() + `\n${line}\n`;
      });
      new Notice('Captured task to Task Inbox');
    }).open();
  }
};
